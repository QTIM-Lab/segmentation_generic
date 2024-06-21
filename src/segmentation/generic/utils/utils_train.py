import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import MaskFormerImageProcessor, SamProcessor
from typing import Tuple, Any
import torch
import numpy as np
import lightning.pytorch as pl
import math

from src.segmentation.mask2former.loggers.loggers_mask2former import SegmentationPredictionMask2FormerLogger
from src.segmentation.medsam.loggers.loggers_medsam import SegmentationPredictionMedSAMLogger
from src.segmentation.generic.models.models_generic import GenericModel
from src.segmentation.mask2former.data_modules.datasets_mask2former import Mask2FormerDataset
from src.segmentation.medsam.data_modules.datasets_medsam import MedSAMDataset
from src.segmentation.medsam.models.models_medsam import SegmentationMedSAM
from src.segmentation.mask2former.models.models_mask2former import SegmentationMask2Former


def get_model_and_processor(config):
    model: GenericModel
    preprocess = None
    if config.model_arch == 'mask2former':
        preprocess = MaskFormerImageProcessor(ignore_index=0, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)
        model = SegmentationMask2Former(
            configs=dict(config),
            num_classes=1,
            preprocessor=preprocess
        )
    elif config.model_arch == 'medsam':
        preprocess = SamProcessor.from_pretrained("flaviagiammarino/medsam-vit-base")
        model = SegmentationMedSAM(
            configs=dict(config),
            preprocessor=preprocess
        )

    return model, preprocess


# Maybe should be utils_data?
# TODO: fix
def get_dataloader_from_csv(model_arch, csv_path, csv_img_path_col, csv_label_path_col,
                            image_root_dir, label_root_dir, transform, preprocessor,
                            batch_size, num_workers, label_bbox_option):
    data_df = pd.read_csv(csv_path)

    image_paths = data_df[csv_img_path_col].tolist()
    mask_paths = data_df[csv_label_path_col].tolist() if csv_label_path_col is not None else None

    data_set: Dataset[Tuple[torch.Tensor, torch.Tensor, np.ndarray[Any, Any], np.ndarray[Any, Any]]]
    if model_arch == 'mask2former':
        data_set = Mask2FormerDataset(
            image_paths,
            mask_paths,
            transform=transform,
            preprocess=preprocessor,
            image_root_dir=image_root_dir,
            label_root_dir=label_root_dir
        )
    elif model_arch == 'medsam':
        data_set = MedSAMDataset(
            image_paths,
            mask_paths,
            transform=transform,
            preprocess=preprocessor,
            image_root_dir=image_root_dir,
            label_root_dir=label_root_dir,
            label_bbox_option=label_bbox_option
        )

    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return data_loader


def get_dataloaders(config, train_transform, val_transform, preprocess, frac_num=100):
    data_root_dir = config.data_dir

    # get paths
    image_root_dir = data_root_dir + 'images'
    label_root_dir = data_root_dir + 'labels'
    # get csvs
    # train_csv = data_root_dir + f'csvs/miccai_nj/train_{frac_num}.csv'
    # val_csv = data_root_dir + f'csvs/miccai_nj/val_{frac_num}.csv'
    # test_csv = data_root_dir + 'csvs/miccai_nj/test_bb.csv'
    train_csv = data_root_dir + f'csvs/train_{frac_num}.csv'
    val_csv = data_root_dir + f'csvs/val_{frac_num}.csv'
    # val_csv = data_root_dir + f'csvs/val_100.csv'
    test_csv = data_root_dir + 'csvs/test_bb.csv'

    # get column names
    # TODO: fix hardcode?
    csv_img_path_col = 'image'
    csv_label_path_col = 'mask'
    # import pdb; pdb.set_trace()
    train_dataloader = get_dataloader_from_csv(
        # config.label_bbox_option
        model_arch=config.model_arch,
        csv_path=train_csv,
        csv_img_path_col=csv_img_path_col,
        csv_label_path_col=csv_label_path_col,
        image_root_dir=image_root_dir,
        label_root_dir=label_root_dir,
        transform=train_transform,
        preprocessor=preprocess,
        batch_size=config.gpu_max_batch_size,
        num_workers=config.num_workers,
        label_bbox_option=config.label_bbox_option
    )

    val_dataloader = get_dataloader_from_csv(
        model_arch=config.model_arch,
        csv_path=val_csv,
        csv_img_path_col=csv_img_path_col,
        csv_label_path_col=csv_label_path_col,
        image_root_dir=image_root_dir,
        label_root_dir=label_root_dir,
        transform=val_transform,
        preprocessor=preprocess,
        batch_size=config.gpu_max_batch_size,
        num_workers=config.num_workers//2,
        label_bbox_option=config.label_bbox_option
    )

    test_dataloader = get_dataloader_from_csv(
        model_arch=config.model_arch,
        csv_path=test_csv,
        csv_img_path_col=csv_img_path_col,
        csv_label_path_col=csv_label_path_col,
        image_root_dir=image_root_dir,
        label_root_dir=label_root_dir,
        transform=val_transform,
        preprocessor=preprocess,
        batch_size=config.gpu_max_batch_size,
        num_workers=config.num_workers//2,
        label_bbox_option=config.label_bbox_option
    )

    return train_dataloader, val_dataloader, test_dataloader


def get_first_n_batches(input_dataloader, n=5):
    # Initialize an empty list to store the first 5 batches
    first_n_batches = []

    # Use enumerate to iterate over the DataLoader with an index
    for batch_idx, batch in enumerate(input_dataloader):
        if batch_idx < n:
            first_n_batches.append(batch)
        else:
            break  # Stop after getting the first n

    return first_n_batches


def get_segmentation_callback(model_arch, batches, processor):
    segmentation_callback: pl.callbacks.Callback
    if model_arch == 'mask2former':
        segmentation_callback = SegmentationPredictionMask2FormerLogger(
            val_samples=batches,
            preprocessor=processor
        )
    elif model_arch == 'medsam':
        segmentation_callback = SegmentationPredictionMedSAMLogger(
            val_samples=batches,
            preprocessor=processor,
            infer_gpu=2
        )
    else:
        return None
    return segmentation_callback


def get_sweep_config(model_arch, train_params, system_params, gpu_id):
    return {
        'method': train_params['sweep_method'],
        'name': train_params['wb_sweepname'],
        'metric': {
            'goal': train_params['sweep_goal'],
            'name': train_params['sweep_metric_name']
        },
        'parameters': {
            # Parameters (Model arch)
            'model_arch': {'values': [model_arch]},
            # Frac num
            'frac_num': train_params['frac_num'],
            # Hyperparameters (ones with log need manual setting)
            'batch_size': train_params['batch_size'],
            # 'lr': {
            #     'max': math.log(train_params['lr']['max']),
            #     'min': math.log(train_params['lr']['min']),
            #     'distribution': train_params['lr']['distribution']
            # },
            'lr': train_params['lr'],
            'grad_clip_val': train_params['grad_clip_val'],
            'augmentations': train_params['augmentations'],
            'patience': train_params['patience'],
            'max_epochs': train_params['max_epochs'],
            'early_stopping_monitor': train_params['early_stopping_monitor'],
            'optimizer_name': train_params['optimizer_name'],
            'scheduler_name': train_params['scheduler_name'],
            'adamw_weight_decay': {
                'max': math.log(train_params['adamw_weight_decay']['max']),
                'min': math.log(train_params['adamw_weight_decay']['min']),
                'distribution': train_params['adamw_weight_decay']['distribution']
            },
            'sgd_momentum': {
                'max': math.log(train_params['sgd_momentum']['max']),
                'min': math.log(train_params['sgd_momentum']['min']),
                'distribution': train_params['sgd_momentum']['distribution']
            },
            'log_every_n_steps': train_params['log_every_n_steps'],
            # System
            'data_dir': {'values': [system_params['data_dir']]},
            'output_dir': {'values': [system_params['output_dir']]},
            'num_workers': {'values': [system_params['num_workers']]},
            'gpu_max_batch_size': {'values': [system_params['gpu_max_batch_size']]},
            'gpu_id': {'values': [gpu_id]},
            'label_bbox_option': train_params['label_bbox_option'],
        },
    }
