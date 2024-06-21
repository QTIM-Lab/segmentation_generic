#!/bin/sh

python src/segmentation/generic/run/run_infer.py \
    --all_models_root_dir /sddata/data/geographic_atrophy/output_logger/label \
    --config_augmentations_path /sddata/projects/segmentation_generic/yamls/augmentations/medsam/high_augs.yaml \
    --holdout_csv_path /sddata/data/geographic_atrophy/nj_110/csvs/test_bb.csv \
    --model_arch medsam \
    --image_root_dir /sddata/data/geographic_atrophy/nj_110/images \
    --label_root_dir /sddata/data/geographic_atrophy/nj_110/labels \
    --num_workers 4 \
    --gpu_id 3 \
    --label_bbox_option label
