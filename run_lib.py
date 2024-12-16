# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import os
import blobfile as bf

import numpy as np
import logging
import random
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
# import evaluation
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
from fft_fn import fft2, ifft2, mask_fn, center_mask
from scipy import io as sio
from datasets import list_mat_files_recursively
from tqdm import tqdm
### BSA - liver
from bsa.networks import Unet as Unet_bsa
from bsa.options import Options as Options_bsa
### CycleMedGAN - brain
from cycle.networks import Unet as Unet_cycle
from cycle.options import Options as Options_cycle

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  torch.cuda.set_device(0)
  sample_dir = os.path.join(workdir, "samples")
  os.makedirs(sample_dir, exist_ok=True)

  tb_dir = os.path.join(workdir, "tensorboard")
  writer = tensorboard.SummaryWriter(tb_dir)

  # Initialize model.
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  os.makedirs(checkpoint_dir, exist_ok=True)
  os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  # state['model'] = nn.DataParallel(state['model'])
  initial_step = int(state['step'])

  # Build data iterators
  train_dl = datasets.get_dataset(config, 'train')
  eval_dl = datasets.get_dataset(config, 'eval')
  train_iter = iter(train_dl)
  eval_iter = iter(eval_dl)
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.training.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))

  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch = next(train_iter).to(config.device).float()
    batch = scaler(batch)
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = next(eval_iter).to(config.device).float()
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        os.makedirs(this_sample_dir, exist_ok=True)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        np.save(os.path.join(this_sample_dir, "sample.np"), sample)
        save_image(image_grid, os.path.join(this_sample_dir, "sample.png"))


def sampling_motion(config, workdir, sample_folder="motion_correction"):
  torch.cuda.set_device(0)
  torch.manual_seed(42)
  np.random.seed(42)
  random.seed(42)
  sample_dir = os.path.join(workdir, sample_folder)
  os.makedirs(sample_dir, exist_ok=True)

  eval_dl = datasets.get_dataset(config, 'eval')
  eval_iter = iter(eval_dl)

  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  score_model = mutils.create_model(config)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  if config.training.sde.lower() == 'vpsde':
    sde = sde_lib.VPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = sde_lib.subVPSDE(beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  continuous = config.training.continuous
  perturbed_fn = losses.get_perturbed_data(sde, continuous)
  sampling_shape = (config.eval.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  file_names = list_mat_files_recursively(bf.join(config.data.data_dir, 'eval'))

  num_sampling_steps = len(file_names)

  logging.info("Starting sampling.")
  ckpt = config.eval.end_ckpt
  ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt}.pth')
  state = restore_checkpoint(ckpt_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  if config.sampling.nn_init == 'bsa':
    opt = Options_bsa().parse()
    pre_model = Unet_bsa(1, 1, opt).to(config.device)
    checkpoint = torch.load(os.path.join(opt.save_path, 'model.pth'), map_location=config.device)
    pre_model.load_state_dict(checkpoint['model'])
    pre_model.eval()
    mask_model = mask_fn(config.sampling.mask)
  elif config.sampling.nn_init == 'cycle':
    opt = Options_cycle().parse()
    pre_model = Unet_cycle(1, 1, opt).to(config.device)
    checkpoint = torch.load(os.path.join(opt.save_path, 'model.pth'), map_location=config.device)
    pre_model.load_state_dict(checkpoint['model'])
    pre_model.eval()
  else:
    raise NotImplementedError(f"NN initialization {config.sampling.nn_init} unknown.")

  for step in tqdm(range(num_sampling_steps)):
    batch = next(eval_iter).to(config.device).float()
    batch = scaler(batch)
    mask_center = center_mask(batch)

    ########################################
    with torch.no_grad():
      if config.sampling.nn_init == 'bsa':
        img = torch.zeros_like(batch)
        for i in range(config.sampling.n_mask):
          mask = mask_model(batch, config.sampling.R)
          inp = torch.real(ifft2(fft2(batch) * mask))
          img += pre_model(inp)
        img = img / config.sampling.n_mask
      elif config.sampling.nn_init == 'cycle':
        img = pre_model(batch)

      for i in range(config.sampling.M):
        x = perturbed_fn(img, config.sampling.t0)
        img, _ = sampling_fn(score_model, x=x, y=batch, mask_center=mask_center, lambd=config.sampling.lambda_consistency)

      this_sample_dir = os.path.join(sample_dir, file_names[step].split('eval/')[1].split('/')[0])
      os.makedirs(this_sample_dir, exist_ok=True)
      img = np.squeeze(img.permute(0, 2, 3, 1).cpu().numpy())
      file_name = file_names[step].split('eval/')[1].split('/')[1]
      sio.savemat(os.path.join(this_sample_dir, file_name), {'img': img})
    ########################################
