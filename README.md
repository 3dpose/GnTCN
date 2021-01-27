# GnTCN

[Graph and Temporal Convolutional Networks for 3D Multi-person Pose Estimation in Monocular Videos](https://arxiv.org/pdf/2012.11806.pdf)

## Installation

Install the latest version of pytorch. (This repo is tested on pytorch 1.3 - 1.7)

Install torchsul
```
pip install --upgrade torchsul 
```

Download the necessary files [here](https://www.dropbox.com/s/3ml0s7wfz57z3oq/tgcn_data.zip?dl=0), and unzip to this project's directory.

## Usage

#### Evaluate on H36M ground truth

```
python eval_gt_h36m.py
```

## Citation

This repository contains the code and models for the following paper. 

> Graph and Temporal Convolutional Networks for 3D Multi-person Pose Estimation in Monocular Videos  
> Cheng Yu, Bo Wang, Bo Yang, Robby T. Tan  
> AAAI Conference on Artificial Intelligence, AAAI 2021.
