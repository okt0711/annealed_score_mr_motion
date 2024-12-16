# Annealed Score-Based Diffusion Model for MR Motion Artifact Reduction

This repository is the official pytorch implementation of "Unpaired MR Motion Artifact Deep Learning Using Outlier-Rejecting Bootstrap Aggregation".  
The code was modified from [[this repo]](https://github.com/yang-song/score_sde_pytorch)

> "Annealed Score-Based Diffusion Model for MR Motion Artifact Reduction",  
> Gyutaek Oh, Sukyoung Jung, Jeong Eun Lee, and Jong Chul Ye,  
> IEEE TCI, 2023 [[Paper]](https://ieeexplore.ieee.org/document/10375761)

## Requirements

The code is implented in Python 3.7 with below packages.
```
torch               1.8.1
numpy               1.21.6
scipy               1.7.3
ml-collections      0.1.1
tensorflow          2.4.0
```

## Project Structure
```
├── models
│   ├── ...
├── op
│   ├── ...
├── configs
│   ├── ve
│   │   ├── ncsnpp_continuous.py
│   │   └── ncsnpp_continuous_sampling.py
│   ├── default_configs.py
│   └── sampling_configs.py
├── bsa
│   ├── ...
├── cycle
│   ├── ...
├── ...
├── datasets.py
├── fft_fn.py
├── sde_lib.py
├── sampling.py
├── run_lib.py
└── main.py
```
1. ```models```, ```op```: contain models and CUDA kernels
2. ```configs```: contains configurations for training, sampling, data, model, etc.
3. ```bsa```, ```cycle```: contain models of "bootstrap subsampling and aggregation" and "CycleMedGAN-V2.0" for "NN init".
4. ```datasets.py```: contains helper functions for pre/post-processing of data.
5. ```fft_fn.py```: contains helper functions for Fourier transforms.
6. ```sde_lib.py```: contains the definition of SDEs (e.g. VE-SDE)
7. ```sampling.py```: contains Algorithm 2 of the paper.
8. ```run_lib.py```: contains training and sampling functions.
9. ```main.py```: contains maing script for training or sampling.

## Pre-Trained Models
You can download pre-trained models from [[here]](https://drive.google.com/drive/folders/1N7-zXjMn7HfvNkEhPfSL21MMupOKweR6?usp=sharing).
The folder contains pre-trained diffusion models with brain or liver MR images.
Also, you can download "bootstrap subsampling and aggregation"(bsa) and "CycleMedGAN-V2.0"(cycle) for "NN init".

## Training and Sampling
You can train your model from scratch by running with:
```
python main.py --mode train
               --config ./configs/ve/ncsnpp_continuous.py
               --eval_folder /your/folder/name
               --workdir /your/work/directory
```
You can generate samples by running with:
```
python main.py --mode eval
               --config ./configs/ve/ncsnpp_continuous_sampline.py
               --eval_folder /your/folder/name
               --workdir /your/work/directory
```

## Citation
If you find our work interesting, please consider citing
```
@article{oh2023annealed,
  title={Annealed score-based diffusion model for mr motion artifact reduction},
  author={Oh, Gyutaek and Jung, Sukyoung and Lee, Jeong Eun and Ye, Jong Chul},
  journal={IEEE Transactions on Computational Imaging},
  year={2023},
  publisher={IEEE}
}
```
