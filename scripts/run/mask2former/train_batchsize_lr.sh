#!/bin/sh

python src/segmentation/generic/run/run_train.py \
    --model_arch mask2former \
    --train_yaml /home/kindersc/repos/segmentation_generic/src/segmentation/generic/run/yamls/training/sweeps/mask2former/sweep_hyperparams.yaml \
    --system_yaml /home/kindersc/repos/segmentation_generic/src/segmentation/generic/run/yamls/system/lt2.yaml \
    --gpu_id 0