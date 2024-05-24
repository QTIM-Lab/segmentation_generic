import wandb
import lightning.pytorch as pl
import numpy as np
import torch
from monai.losses.dice import DiceLoss


class SegmentationPredictionMedSAMLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, preprocessor=None, num_samples=10, infer_gpu=-1):
        super().__init__()
        self.num_samples = num_samples
        self.preprocessor = preprocessor
        self.val_samples = val_samples
        self.metric = DiceLoss(sigmoid=False, squared_pred=False, reduction="mean")
        self.infer_gpu = infer_gpu

    def get_pixel_mask(self, label_mask):
        # Convert the PyTorch tensor to a NumPy array
        label_mask_np = label_mask.cpu().numpy()

        # Create a new NumPy array with the desired shape (batch_size, height, width)
        new_mask = np.zeros((label_mask_np.shape[0], label_mask_np.shape[2], label_mask_np.shape[3]), dtype=np.uint8)

        # Iterate over the batch dimension
        for batch in range(label_mask_np.shape[0]):
            # Set the pixel values based on the channel values
            new_mask[batch][label_mask_np[batch, 0] == 1] = 0
            new_mask[batch][label_mask_np[batch, 1] == 1] = 1

        return new_mask

    def on_validation_epoch_end(self, trainer, pl_module):
        for sample in self.val_samples:
            # Get batch items, on CPU to start
            pixel_values, mask_labels, original_sizes, reshaped_input_sizes, input_boxes = sample

            # Put batch items on device
            sample = SegmentationPredictionMedSAMLogger.move_batch_to_device(
                batch=sample,
                device=pl_module.device,
            )

            # Infer
            output = pl_module(sample)

            # Convert batched tensors to list of tensors (b,h,w) -> [(h,w), ...]
            pixel_values_list = torch.unbind(pixel_values, dim=0)
            mask_labels_list = torch.unbind(mask_labels, dim=0)

            # Convert to predictions
            preds_list = self.preprocessor.image_processor.post_process_masks(
                output.pred_masks.sigmoid().cpu(),
                original_sizes,
                reshaped_input_sizes,
                binarize=False
            )

            # first [0] because I cant fit a batch size of 2 to see if it is batch
            # or if list returns multiple...
            # second [0] for class dim for metric calculation, always first class
            # and convert to integers instead of bool mask
            preds_list = [torch.round(pred[0][0].detach().cpu()) for pred in preds_list]

            # Get a list with the loss values calculated for reporting
            losses_list = [1 - self.metric(pred, mask_label).item() for pred, mask_label in zip(preds_list, mask_labels_list)]

            # get class labels
            class_lbls = {
                0: 'background',
                1: 'geographic-atrophy',
            }

            # log prediction and ground truth segmentations for images in batch
            trainer.logger.experiment.log({
                "examples": [
                    wandb.Image(img.cpu().numpy().transpose(1, 2, 0), caption=f"dice:{loss:.3f}", masks={
                        "predictions": {
                            "mask_data": pred.numpy(),
                            "class_labels": class_lbls
                        },
                        "ground_truth": {
                            "mask_data": mask.numpy(),
                            "class_labels": class_lbls
                        }

                    })
                    for img, mask, pred, loss in zip(pixel_values_list, mask_labels_list, preds_list, losses_list)
                ]
            })

    @staticmethod
    def move_batch_to_device(batch, device):
        return tuple(item.to(device) for item in batch)
