import argparse
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import yaml

from src.segmentation.generic.augmentations.augmentations_train_val import get_train_and_val_transform
from src.segmentation.generic.utils.utils_train import (
    get_dataloaders,
    get_first_n_batches,
    get_model_and_processor,
    get_segmentation_callback,
    get_sweep_config
)


def parse_args():
    parser = argparse.ArgumentParser(description="Segmentation")
    parser.add_argument('--model_arch', type=str, choices=['mask2former', 'medsam'], required=True, help="Model architecture")
    parser.add_argument('--train_yaml', type=str, required=True, help="Path to the train input file")
    parser.add_argument('--system_yaml', type=str, required=True, help="Path to the system input file")
    parser.add_argument('--gpu_id', type=int, default=0, help="Which GPU to run on (distributed not available yet)")
    parser.add_argument('--verbose', action='store_true', help="Increase output verbosity")

    return parser.parse_args()


def train_model(config=None):
    with wandb.init():
        config = wandb.config  # I think this is {...parameters}

        print('using: ', config.frac_num)
        # config.method

        # Initialize Augmentations
        with open(config.augmentations, 'r') as file:
            augmentation_params = yaml.safe_load(file)
        train_transform, val_transform = get_train_and_val_transform(augmentation_params)

        # Initialize Model and Processor (preprocessor?)
        model, preprocess = get_model_and_processor(config)

        # Get dataset
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
            config=config,
            train_transform=train_transform,
            val_transform=val_transform,
            preprocess=preprocess,
            frac_num=config.frac_num
        )

        # and first n batches for callback function that needs them
        first_n_batches = get_first_n_batches(input_dataloader=val_dataloader, n=10)

        # Initialize Callbacks
        segmentation_callback = get_segmentation_callback(
            model_arch=config.model_arch,
            batches=first_n_batches,
            processor=preprocess
        )
        early_stop_callback = pl.callbacks.EarlyStopping(monitor=config.early_stopping_monitor, patience=config.patience)
        checkpoint_callback = pl.callbacks.ModelCheckpoint()
        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

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
            callbacks=[early_stop_callback, segmentation_callback, checkpoint_callback, lr_monitor],
            devices=[config.gpu_id],
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
    model_arch = args.model_arch
    train_yaml = args.train_yaml
    system_yaml = args.system_yaml
    gpu_id = args.gpu_id

    with open(train_yaml, 'r') as file:
        train_params = yaml.safe_load(file)

    with open(system_yaml, 'r') as file:
        system_params = yaml.safe_load(file)

    sweep_config = get_sweep_config(
        model_arch=model_arch,
        train_params=train_params,
        system_params=system_params,
        gpu_id=gpu_id
    )

    sweep_id = wandb.sweep(sweep_config, project=train_params['wb_project'])
    wandb.agent(sweep_id=sweep_id, function=train_model, count=train_params['sweep_count'])
