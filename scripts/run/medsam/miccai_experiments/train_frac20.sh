#!/bin/sh

python src/segmentation/generic/run/run_train.py \
    --model_arch medsam \
    --train_yaml /home/kindersc/repos/segmentation_generic/yamls/training/sweeps/medsam/miccai_experiments/experiment_20.yaml \
    --system_yaml /home/kindersc/repos/segmentation_generic/yamls/system/lt2.yaml \
    --gpu_id 0