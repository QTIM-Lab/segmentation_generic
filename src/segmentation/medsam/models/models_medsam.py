from transformers import SamModel
from monai.losses.dice import DiceLoss

from src.segmentation.generic.models.models_generic import GenericModel


class SegmentationMedSAM(GenericModel):
    '''
    Credit to Weights and Biases
    '''
    def __init__(self, configs, preprocessor):
        super().__init__(configs=configs)

        self.model = SamModel.from_pretrained("flaviagiammarino/medsam-vit-base")
        self.preprocessor = preprocessor
        self.metric = DiceLoss(sigmoid=False, squared_pred=True, reduction="mean")

    # will be used during inference
    def forward(self, x):
        pixel_values, mask_labels, original_sizes, reshaped_input_sizes, input_boxes = x
        output = self.model(
            pixel_values=pixel_values,
            original_sizes=original_sizes,
            reshaped_input_sizes=reshaped_input_sizes,
            input_boxes=input_boxes,
            multimask_output=False
        )
        return output

    def common_step(self, batch, batch_idx):
        pixel_values, mask_labels, original_sizes, reshaped_input_sizes, input_boxes = batch

        # Forward pass
        outputs = self(batch)

        probs = self.preprocessor.image_processor.post_process_masks(
            outputs.pred_masks.sigmoid().cpu(),
            original_sizes.cpu(),
            reshaped_input_sizes.cpu(),
            binarize=False
        )

        probs = probs[0]  # naughy direct access *shrug*
        probs = probs.squeeze(1)  # squeeze off class dim for metric calculation

        loss = self.metric(probs, mask_labels.cpu())
        metric_value = 1 - loss.item()

        return loss, metric_value
