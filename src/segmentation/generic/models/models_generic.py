import lightning.pytorch as pl
import torch
from typing import Optional
from monai.losses.dice import DiceLoss # import also DiceMetric


class GenericModel(pl.LightningModule):
    '''
    Credit to Weights and Biases
    '''
    def __init__(self, configs):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = configs['lr']
        self.optimizer_name = configs['optimizer_name']
        self.scheduler_name = configs['scheduler_name']
        self.adamw_weight_decay = configs['adamw_weight_decay']
        self.sgd_momentum = configs['sgd_momentum']
        self.metric = DiceLoss(sigmoid=False, squared_pred=True, reduction="mean")
        # squared_pred, DiceMetrics vs DiceLoss


    # will be used during inference
    def forward(self, x):
        return x

    def common_step(self, batch, batch_idx):
        loss = 0
        metric_value = 0
        

        return loss, metric_value
    
    # def get_probabilities(self, batch, batch_idx)::
        # prob = 0
        # return prob

    def training_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=False, on_epoch=True, logger=True)
        self.log('train_acc', metric_value, on_step=False, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', metric_value, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, metric_value = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', metric_value, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer
        optimizer: torch.optim.Optimizer  # Type hint for optimizer
        if self.optimizer_name == 'adam':
            # Default adam settings, only experiment with AdamW decay
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=0)
        elif self.optimizer_name == 'adamw':
            # AdamW uses weight decay
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.adamw_weight_decay)
        elif self.optimizer_name == 'sgd':
            # Define an SGD optimizer with momentum
            optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.sgd_momentum)

        # Scheduler
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        if self.scheduler_name == 'exponential_decay':
            # Exponential decay scheduler
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95,  # Decay rate
            )
        elif self.scheduler_name == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=25,  # Maximum number of iterations
                eta_min=self.learning_rate/50,  # Minimum learning rate
            )
        elif self.scheduler_name == 'cyclic_lr':
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                base_lr=self.learning_rate/25,
                max_lr=self.learning_rate*25,
                step_size_up=100
            )
        else:
            return optimizer

        return [optimizer], [scheduler]
