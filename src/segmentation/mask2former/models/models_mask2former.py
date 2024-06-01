import torch
from transformers import Mask2FormerForUniversalSegmentation
from monai.losses.dice import DiceLoss

from src.segmentation.generic.utils.utils_logging import get_pixel_mask
from src.segmentation.generic.models.models_generic import GenericModel


class SegmentationMask2Former(GenericModel):
    '''
    Credit to Weights and Biases
    '''
    def __init__(self, configs, num_classes, preprocessor):
        super().__init__(configs=configs)

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

        self.metric = DiceLoss(sigmoid=False, squared_pred=True, reduction="mean")

    # will be used during inference
    def forward(self, x):
        pixel_values, mask_labels, class_labels = x
        output = self.model(
            pixel_values=pixel_values,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )
        return output

    def common_step(self, batch, batch_idx):
        pixel_values, mask_labels, class_labels = batch

        # Forward pass
        outputs = self(batch)

        # Backward propagation
        loss = outputs.loss

        # Get segmentation maps
        target_sizes = [(image.shape[1], image.shape[2]) for image in pixel_values]
        predicted_segmentation_maps = self.preprocessor.post_process_semantic_segmentation(
            outputs,
            target_sizes=target_sizes
        )

        # Get the prediction segmentation maps into proper valued format
        predicted_segmentation_maps_tensor = torch.stack(predicted_segmentation_maps)
        predicted_segmentation_maps_tensor[predicted_segmentation_maps_tensor <= 1] = 0
        predicted_segmentation_maps_tensor[predicted_segmentation_maps_tensor == 2] = 1

        # Get ground truth segmentations from mask labels
        # mask_labels = (b, n_classes, h, w) with 1s for classes, we want -> (b, h, w) with classes as pixel values, offset by 1
        gt_seg_maps_tensor = get_pixel_mask(mask_labels, class_offset=0)

        # Calculate mean IoU for just GA (binary on that class)
        mean_dice = 1 - self.metric(predicted_segmentation_maps_tensor, gt_seg_maps_tensor)

        return loss, mean_dice
