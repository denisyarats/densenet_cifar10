# Meta-learning learning rates with higher
This is the code to training a DenseNet on CIFAR10 with meta-learned learning rates via higher.

## Requirements
We assume you have access to a gpu that can run CUDA 9.2. Then, the simplest way to install all required dependencies is to create an anaconda environment by running:
```
conda env create -f conda_env.yml
```
After the instalation ends you can activate your environment with:
```
source activate pytorch_env
```

## Instructions
To start the training script, simply run:
```
python train.py \
    --work_dir ./log \
    --meta_batch_size 16 \
    --meta_num_train_steps 5 \
    --meta_num_test_steps 5 \
    --split_ratio 0.01
```
This will produce 'log' folder, where all the outputs are going to be stored including train/test/meta logs, and tensorboard blobs. One can attach a tensorboard to monitor training by running
```
tensorboard --logdir log
```
and opening up tensorboad in your browser. The console output is also available in a form:
```
| train | T: 100 | E: 1 | D: 11.1 s | L: 2.0694 | A: 24.1094 | LR: 0.0000
```
a training entry decodes as:
```
train - training episode
T - total updates 
E - total number of epochs
D - duration
L - loss
A - accuracy
```
There will be similar entries from test and meta iterations.
