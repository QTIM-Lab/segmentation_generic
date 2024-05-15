import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger

from src.classification.simplenet.data_modules.dm_cifar import CIFAR10DataModule
from src.classification.simplenet.loggers.loggers_image_prediction import ImagePredictionLogger
from src.classification.simplenet.utils.utils_train import get_val_samples
from src.classification.simplenet.models.models_simplenet import SimpleNet


def train_model():
    with wandb.init(project='cifar-sweep', dir='/sddata/projects/wandb_examples/i_dir'):
        config = wandb.config
        # Initialize Callbacks
        early_stop_callback = pl.callbacks.EarlyStopping(monitor="val_loss")
        checkpoint_callback = pl.callbacks.ModelCheckpoint()

        # Initialize data
        dm = CIFAR10DataModule(config)
        val_samples = get_val_samples(dm)

        # # Initialize model
        model = SimpleNet(input_shape=(3, 32, 32), num_classes=dm.num_classes, configs=dict(config))

        # Initialize wandb logger
        wandb_logger = WandbLogger(project='wandb-lightning', job_type='train', log_model='all', save_dir='/sddata/projects/wandb_examples/l_dir')

        # log gradients, parameter histogram and model topology
        wandb_logger.watch(model, log="all")

        # Initialize a trainer
        trainer = pl.Trainer(
            max_epochs=4,
            logger=wandb_logger,
            callbacks=[early_stop_callback, ImagePredictionLogger(val_samples), checkpoint_callback],
            devices=1,
            accelerator='gpu'
        )

        # Run training
        trainer.fit(model, dm)

        # Evaluate the model on the test set
        trainer.test(dataloaders=dm.test_dataloader())


if __name__ == '__main__':
    sweep_config = {
        'method': 'random',
        'name': 'first_sweep',
        'metric': {
            'goal': 'minimize',
            'name': 'validation_loss'
        },
        'parameters': {
            'batch_size': {'values': [16, 24, 32, 48, 64]},
            'lr': {'max': 0.001, 'min': 0.00001}
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="cifar-sweep")
    wandb.agent(sweep_id=sweep_id, function=train_model, count=4)
