# Spatio-Temporal Scene-Graph Embedding for Autonomous Vehicle Collision Prediction

This repository includes the code and dataset information required for reproducing the results in our paper, *Spatio-Temporal Scene-Graph Embedding for Autonomous Vehicle Collision Prediction*. Furthermore, we also integrated the source code of [our baseline method](https://arxiv.org/abs/1711.10453), into this repo. The baseline approach infers the likelihood of a future collision using deep ConvLSTMs. Our approach incoporates both spatial modeling and temporal modeling in the task of collision predition using MRGCN.

For fabricating the lane-changing datasets, we use Carla [CARLA](https://github.com/carla-simulator/carla) 0.9.8 which is an open-source autonomous car driving simulator. We also utilized the [scenario_runner](https://github.com/carla-simulator/scenario_runner) which was designed for CARLA challenge event. For real-driving datasets, we used Honda-Driving Dataset (HDD) in our experiments. 

The scene-graph dataset used in our paper is published [here](http://ieee-dataport.org/3618).

The architecture of this repository is as below:
- **sg-risk-assessment/**: this folder consists of all the related source files used for our scene-graph based approach (SG2VEC). 
- **baseline-risk-assessment/**: this folder consists of all the related source files used for the baseline method (DPM).
- **train_sg2vec.py**: the script that triggers our scene-graph based approach. 
- **train_dpm.py**: the script that triggers the baseline model.

# To Get Started
We recommend our potential users to use [Anaconda](https://www.anaconda.com/) as the primary virtual environment. The requirements to run through our repo are as follows,
- python >= 3.6 
- torch == 1.6.0
- torch_teometric == 1.6.1

First, download and install Anaconda here:
https://www.anaconda.com/products/individual

If you are using a GPU, install the corresponding CUDA toolkit for your hardware from Nvidia here:
https://developer.nvidia.com/cuda-toolkit

Next, create a conda virtual environment running Python 3.6:
```shell
conda create --name av python=3.6
```

After setting up your environment. Activate it with the following command:

```shell
conda activate av
```

Install PyTorch to your conda virtual environment by following the instructions here for your CUDA version:
https://pytorch.org/get-started/locally/

In our experiments we used Torch 1.5 and 1.6 but later versions should also work fine.

Next, install the PyTorch Geometric library by running the corresponding commands for your Torch and CUDA version:
https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html

Once this setup is completed, install the rest of the requirements from requirements.txt:

```shell
pip install -r requirements.txt
```
---

# Usages
For running the sg-collision-prediction in this repo, you may refer to the following commands:
```shell
$ python train_sg2vec.py --pkl_path risk-assessment/scenegraph/synthetic/271_dataset.pkl

# --pkl_path + [wherever path that stores the downloaded pkl]
# For tuning hyperparameters view the config class of sg2vec_trainer.py
```

For running the baseline-risk-assessment in this repo, you may refer to the following commands:
```shell
$ python train_dpm.py --load_pkl True --pkl_path risk-assessment/scene/synthetic/271_dataset.pkl

# --pkl_path + [wherever path that stores the downloaded pkl]
# For tuning hyperparameters view the config class of dpm_trainer.py
```

After running these commands, the expected outputs are a dump of metrics logged by wandb:
```shell
wandb:                    train_recall ▁████████████████████
wandb:                   val_precision █▁▅▄▅▄▆▆▆▅▄▄▇▆▅▆▅▇▆▆▆
wandb:                      val_recall ▁████████████████████
wandb:                       train_fpr ▁█▅▅▄▅▃▂▂▂▂▂▂▂▂▂▂▂▂▂▂
wandb:                       train_tnr █▁▄▅▅▅▆▇▇▇▇▇▇▇▇▇▇▇▇▇▇
wandb:                       train_fnr █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                         val_fpr ▁█▄▅▄▅▃▃▃▄▄▅▂▃▃▃▄▂▃▃▃
wandb:                         val_tnr █▁▆▄▆▄▆▆▆▆▅▄▇▆▆▆▆▇▆▆▆
wandb:                         val_fnr █▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                      best_epoch ▁▁▂▂▂▂▃▃▄▄▄▄▅▅▅▅▅▇▇▇█
wandb:                   best_val_loss █▃▂▂▂▂▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:                    best_val_acc ▁▆█▇█████████████████
wandb:                    best_val_auc ▁▅▆▆▇▇▇▇████▇▇▇▇▇████
wandb:                    best_val_mcc ▁▇███████████████████
wandb:           best_val_acc_balanced ▁████████████████████
wandb:                       train_mcc ▁▇▇▇▇▇███████████████
wandb:                         val_mcc ▁▇███████████████████
```

A graphical visualization of the model outputs including loss and additional metrics can be viewed by creating and linking your runs to [wandb](https://wandb.ai/home).