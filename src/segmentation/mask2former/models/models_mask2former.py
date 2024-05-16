import lightning.pytorch as pl
import torch
from transformers import Mask2FormerForUniversalSegmentation
import evaluate
import time
from typing import Optional

from src.segmentation.mask2former.utils.utils_logging import get_pixel_mask


class SegmentationMask2Former(pl.LightningModule):
    '''
    Credit to Weights and Biases
    '''
    def __init__(self, configs, num_classes, preprocessor):
        super().__init__()

        # log hyperparameters
        self.save_hyperparameters()
        self.learning_rate = configs['lr']
        self.optimizer_name = configs['optimizer_name']
        self.scheduler_name = configs['scheduler_name']
        self.adamw_weight_decay = configs['adamw_weight_decay']
        self.sgd_momentum = configs['sgd_momentum']

        # for classes
        # (all need to have the 0 unlabeled and 1 background I think...)
        id2label = {
            0: 'unlabeled',
            1: 'bg'
        }
        for i in range(num_classes):
            id2label[i+2] = 'class' + str(i)
        self.id2label = id2label

        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            'facebook/mask2former-swin-large-cityscapes-semantic',
            id2label=id2label,
            ignore_mismatched_sizes=True
        )

        # Create an empty preprocessor
        self.preprocessor = preprocessor

        # experiment id should be a unique string value for the experiment. can use the output folders name
        self.metric = evaluate.load('mean_iou', experiment_id=str(time.time()))

    # will be used during inference
    def forward(self, pixel_values, mask_labels=None, class_labels=None):
        x = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        return x

    def common_step(self, batch, batch_idx):
        pixel_values, mask_labels, class_labels = batch

        # Forward pass
        outputs = self(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        # Backward propagation
        loss = outputs.loss

        # Get segmentation maps
        target_sizes = [(image.shape[1], image.shape[2]) for image in pixel_values]
        predicted_segmentation_maps = self.preprocessor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes
        )

        # Get ground truth segmentations from mask labels
        # mask_labels = (b, n_classes, h, w) with 1s for classes, we want -> (b, h, w) with classes as pixel values, offset by 1
        ground_truth_segmentation_maps = get_pixel_mask(mask_labels.cpu().numpy(), class_offset=1)

        # Calculate mean IoU between classes (background and GA)
        # Ideally would only do for GA, but other class takes up ignore index,
        # and for reporting can fix, for this is mathematically same?
        mean_iou = self.metric.compute(
            num_labels=len(self.id2label),
            ignore_index=0,
            references=ground_truth_segmentation_maps,
            predictions=predicted_segmentation_maps
        )['mean_iou']

        return loss, mean_iou

    def training_step(self, batch, batch_idx):
        loss, mean_iou = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        self.log('train_acc', mean_iou, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, mean_iou = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', mean_iou, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, mean_iou = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', mean_iou, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Optimizer
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
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
                max_lr=self.learning_rate
            )
        else:
            return optimizer

        return [optimizer], [scheduler]
