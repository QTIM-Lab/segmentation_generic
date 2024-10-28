#!/bin/sh

python src/segmentation/generic/run/run_train.py \
    --model_arch medsam \
    --train_yaml /path/to/repo/segmentation_generic/yamls/training/sweeps/medsam/miccai_experiments/my_example.yaml \
    --system_yaml /path/to/repo/segmentation_generic/yamls/system/my_system.yaml \
    --gpu_id 0
