import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from transformers import MaskFormerImageProcessor
import yaml
import math

from src.segmentation.mask2former.models.models_mask2former import SegmentationMask2Former
from src.segmentation.mask2former.loggers.loggers_segmentation_prediction import SegmentationPredictionLogger
from src.segmentation.mask2former.augmentations.augmentations_train_val import get_train_and_val_transform
from src.segmentation.mask2former.utils.utils_train import get_dataloaders, get_first_n_batches

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument('--train_yaml', type=str, required=True, help="Path to the train input file")
    parser.add_argument('--system_yaml', type=str, required=True, help="Path to the system input file")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")

    return parser.parse_args()


def train_model(config=None):
    with wandb.init():
        config = wandb.config

        # Initialize Callbacks
        early_stop_callback = pl.callbacks.EarlyStopping(monitor=config.early_stopping_monitor, patience=config.patience)
        checkpoint_callback = pl.callbacks.ModelCheckpoint()

        # Initialize Preprocessor
        preprocess = MaskFormerImageProcessor(ignore_index=0, do_reduce_labels=False, do_resize=False, do_rescale=False, do_normalize=False)

        # Initialize Augmentations
        with open(config.augmentations, 'r') as file:
            augmentation_params = yaml.safe_load(file)
        train_transform, val_transform = get_train_and_val_transform(augmentation_params)

        # Get dataset
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config=config,
            train_transform=train_transform,
            val_transform=val_transform,
            preprocess=preprocess
        )

        first_n_batches = get_first_n_batches(input_dataloader=val_dataloader, n=5)

        # Initialize model
        model = SegmentationMask2Former(
            configs=dict(config),
            num_classes=1,
            preprocessor=preprocess
        )

        # Get the run id
        wandb_run = wandb.run
        wandb_run_id = str(wandb_run.id) if wandb_run else 'tmp'

        # Initialize wandb logger
        wandb_logger = WandbLogger(
            project='wandb-lightning',
            job_type='train',
            # log_model='all',
            save_dir=config.output_dir + wandb_run_id
        )

        # log gradients, parameter histogram and model topology
        wandb_logger.watch(model, log="all")

        # Initialize a trainer
        trainer = pl.Trainer(
            max_epochs=config.max_epochs,
            logger=wandb_logger,
            callbacks=[early_stop_callback, SegmentationPredictionLogger(first_n_batches, preprocessor=preprocess), checkpoint_callback],
            devices=1,
            accelerator='gpu',
            log_every_n_steps=config.log_every_n_steps,
            accumulate_grad_batches=config.batch_size//config.gpu_max_batch_size,
            gradient_clip_val=config.grad_clip_val
        )

        # Run training
        trainer.fit(model, train_dataloader, val_dataloader)

        # Evaluate the model on the test set
        trainer.test(dataloaders=test_dataloader)

        # Clean up
        wandb.finish()


if __name__ == '__main__':
    args = parse_args()
    train_yaml = args.train_yaml
    system_yaml = args.system_yaml

    with open(train_yaml, 'r') as file:
        train_params = yaml.safe_load(file)

    with open(system_yaml, 'r') as file:
        system_params = yaml.safe_load(file)

    sweep_config = {
        'method': train_params['sweep_method'],
        'name': train_params['wb_sweepname'],
        'metric': {
            'goal': train_params['sweep_goal'],
            'name': train_params['sweep_metric_name']
        },
        'parameters': {
            # Hyperparameters (ones with log need manual setting)
            'batch_size': train_params['batch_size'],
            'lr': {
                'max': math.log(train_params['lr']['max']),
                'min': math.log(train_params['lr']['min']),
                'distribution': train_params['lr']['distribution']
            },
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
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=train_params['wb_project'])
    wandb.agent(sweep_id=sweep_id, function=train_model, count=train_params['sweep_count'])
