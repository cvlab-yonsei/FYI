# FYI: Flip Your Images for Dataset Distillation
This repository is an official implementation of the paper "FYI: Flip Your Images for Dataset Distillation" using PyTorch.

For detailed information, please refer to our [project site](https://cvlab.yonsei.ac.kr/projects/FYI/) or [paper](https://arxiv.org/abs/2407.08113).

## Requirements
Please install packages in the requirements.

## Getting started
- Prepare the dataset in `/dataset` directory for experiments on `Tiny-ImageNet` and `ImageNet subset`.
- Run the provided scripts as below. Modify the scripts for different settings.
### DC, DSA, and DM
```
cd DC_DSA_DM

# DM for 10 IPC on CIFAR-10
./scripts/dm.sh

# DSA for 10 IPC on CIFAR-10
./scripts/dsa.sh
```
### MTT
```
cd MTT

# Generating teacher trajectories on CIFAR-10 using ZCA whitening
./scripts/buffer.sh

# MTT for 1 IPC on CIFAR-10
./scripts/distill.sh
```

## Acknowledgement
This implementation is built on [DC/DSA/DM](https://github.com/VICO-UoE/DatasetCondensation) and [MTT](https://github.com/georgecazenavette/mtt-distillation). We also applied our method to [IDC](https://github.com/snu-mllab/efficient-dataset-condensation) and [FTD](https://github.com/AngusDujw/FTD-distillation) in our paper. We thank the authors and contributors of these projects.

## Citation
If you find our work useful for your research, please cite our paper:
```
@inproceedings{son2024fyi,
  title={{FYI}: Flip Your Images for Dataset Distillation},
  author={Son, Byunggwan and Oh, Youngmin and Baek, Donghyeon and Ham, Bumsub},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2024}
}
```
