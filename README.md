# GnTCN
[![arXiv](https://img.shields.io/badge/arXiv-2012.11806v3-00ff00.svg)](https://arxiv.org/pdf/2012.11806v3.pdf)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-and-temporal-convolutional-networks-for/3d-multi-person-pose-estimation-absolute-on)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-absolute-on?p=graph-and-temporal-convolutional-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-and-temporal-convolutional-networks-for/3d-multi-person-pose-estimation-root-relative)](https://paperswithcode.com/sota/3d-multi-person-pose-estimation-root-relative?p=graph-and-temporal-convolutional-networks-for)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-and-temporal-convolutional-networks-for/root-joint-localization-on-human3-6m)](https://paperswithcode.com/sota/root-joint-localization-on-human3-6m?p=graph-and-temporal-convolutional-networks-for)

## Introduction

This repository contains the code and models for the following paper. 

> [Graph and Temporal Convolutional Networks for 3D Multi-person Pose Estimation in Monocular Videos](https://arxiv.org/pdf/2012.11806v3.pdf)  
> Cheng Yu, Bo Wang, Bo Yang, Robby T. Tan  
> AAAI Conference on Artificial Intelligence, AAAI 2021.

<p align="center"><img src="Results_on_wild_videos.png" width="86%" alt="" /></p>

### Updates

- 06/07/2021 evaluation code (PCK_abs camera-centric) and pre-trained model for MuPoTS dataset tested and released
- 04/30/2021 evaluation code (PCK person-centric), pre-trained model, and estimated 2D joints for MuPoTS dataset released


## Installation

### Dependencies
[Pytorch](https://pytorch.org/) >= 1.3<br>
Python >= 3.6<br>

Create an enviroment. 
```
conda create -n gntcn python=3.6
conda activate gntcn
```
Install the latest version of pytorch (tested on pytorch 1.3 - 1.7) based on your OS and GPU driver installed following [install pytorch](https://pytorch.org/). For example, command to use on Linux with CUDA 11.0 is like:
```
conda install pytorch torchvision cudatoolkit=11.0 -c pytorch
```

Install opencv-python, torchsul, tqdm, and scipy to run the evaluation code
```
pip install opencv-python
pip install --upgrade torchsul 
pip install tqdm
pip install scipy
```

## Pre-trained Model

Download the pre-trained model and processed human keypoint files (H36M and MuPoTS) [here](https://www.dropbox.com/s/havjgrkaozjyb1k/tgcn_data.zip?dl=0), and unzip the downloaded zip file to this project's directory, two folders and one pkl file are expected to see after doing that (i.e., `./ckpts`, `./mupots`, and `points_eval.pkl`).


## Usage

### MuPoTS dataset evaluation

MuPoTS eval set is needed to perform evaluation, which is available on the [MuPoTS dataset website](http://gvv.mpi-inf.mpg.de/projects/SingleShotMultiPerson/) (download the mupots-3d-eval.zip file, unzip it, and run `get_mupots-3d.sh` to download the dataset).

#### Run evaluation on MuPoTS dataset with estimated 2D joints as input 

The estimated 2D points are included in the data package as well. To evaluate the person-centered poses:
```
python calculate_mupots_detect.py
python eval_mupots.py
```
After running above command, the following PCK (person-centric, pelvis-based origin) value is expected, which matches the number reported in Table 3, PCK 87.5 (percentage) in the paper. 
```
...
PCK_MEAN: 0.8764509703036868
```

To evaluate the poses in camera coordinates (PCK(abs)):
```
python calculate_mupots_detect.py
python calculate_mupots_depth.py
python eval_mupots_dep.py
```
After running above command, the following PCK_abs (camera-centric) value is expected, which matches the number reported in Table 3, PCK_abs 45.7 (percentage) in the paper. 
```
...
PCK_MEAN: 0.45785827181758376
```

#### Run evaluation on MuPoTS dataset with 2D Ground-truth joints as input 

The Ground-truth 2D joints are included in the data package as well to demonstrate the upper-bound performance of the model, where the 2D ground-truth keypoints are used as input to mimic the situation that there is no error in 2D pose estimation. To evaluate with GPU:
```
python calculate_mupots_gt.py
python eval_mupots.py
``` 
After running above command, the following PCK (person-centric, pelvis-based origin) value is expected. 
```
...
PCK_MEAN: 0.8985102807603582
```

### Human3.6M dataset evaluation

#### Run evaluation on Human3.6M dataset with 2D Ground-truth joints as input

Similar to the evaluation above where 2D ground-truth keypoints are used for MuPoTS. The following evaluation code takes 2D Ground-truth joints of the Human3.6M as input to simulate the situation when there is no error in 2D pose estimator, how the proposed method performs. Please note the MPJPE value from this evaluation is lower than the one reported in the paper because the result in Table 5 in the paper was calculated based on the estimated 2D keypoints (i.e., with errors) not from ground-truth. 

If GPU is available and pytorch is installed successfully, the GPU evaluation code can be used,
```
python eval_gt_h36m.py
```
After running above command, the following MPJPE value is expected. 
```
...
MPJPE: 0.0180
```

If GPU is not available or pytorch is not successfully installed, the CPU evaluation code can be used,
```
python eval_gt_h36m_cpu.py
```
Result is the same as the GPU evaluation code. 

#### Testing on wild videos

Please note that we didn't include 2D pose estimator code in this repository to keep it simple, please use off-the-shelf 2D pose estimation methods to get 2D joints first, and together with the code from this repository to infer 3D human pose on testing videos (the TCN takes multiple frames as input). In particular, as stated in the paper: we use the original implementation of [HRNet](https://github.com/HRNet/HRNet-Human-Pose-Estimation) as the 2D pose estimator and extract PAF from [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).

## Citation

If this work is useful for your research, please cite our paper. 
```
@article{cheng2020graph,
  title={Graph and Temporal Convolutional Networks for 3D Multi-person Pose Estimation in Monocular Videos},
  author={Cheng, Yu and Wang, Bo and Yang, Bo and Tan, Robby T},
  journal={AAAI},
  year={2021}
}
```
