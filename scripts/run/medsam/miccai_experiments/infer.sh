#!/bin/sh
# --all_models_root_dir /sddata/data/geographic_atrophy/output_logger/label \
python ./src/segmentation/generic/run/run_infer.py \
    --config_augmentations_path /home/thakuriu/active_learning/segmentation_generic/yamls/augmentations/medsam/low_augs.yaml \
    --holdout_csv_path /sddata/data/geographic_atrophy/nj_110/csvs/test.csv \
    --model_arch medsam \
    --image_root_dir /sddata/data/geographic_atrophy/nj_110/images \
    --label_root_dir /sddata/data/geographic_atrophy/nj_110/labels \
    --num_workers 4 \
    --gpu_id 0 \
    --label_bbox_option label
