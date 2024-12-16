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
"""Return training and evaluation/test datasets from config files."""
import blobfile as bf
import numpy as np
from torch.utils.data import DataLoader, Dataset
from scipy import io as sio
import random
import cv2


def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2. - 1.
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.) / 2.
    else:
        return lambda x: x


def list_mat_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["mat"]:
          results.append(full_path)
        elif bf.isdir(full_path):
          results.extend(list_mat_files_recursively(full_path))
    return results


class ImageDataset(Dataset):
    def __init__(self, image_paths, random_flip=True):
        super().__init__()
        self.local_images = image_paths
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        path = self.local_images[idx]
        with bf.BlobFile(path, "rb") as f:
            data = sio.loadmat(path)
            img = data['img']

            if self.random_flip and random.random() < 0.5:
                img = img[:, ::-1].copy()

        return img[np.newaxis, :, :]


def get_dataset(config, mode):
    assert mode in ['train', 'eval']

    if not config.data.data_dir:
        raise ValueError("Unspecified data directory")
    all_files = list_mat_files_recursively(bf.join(config.data.data_dir, mode))

    if mode == 'train':
        ds = ImageDataset(all_files, config.data.random_flip)
        dl = DataLoader(ds, batch_size=config.training.batch_size, shuffle=True, num_workers=1, drop_last=True)
    else:
        ds = ImageDataset(all_files, False)
        dl = DataLoader(ds, batch_size=config.eval.batch_size, shuffle=False, num_workers=0, drop_last=True)
    while True:
        yield from dl
