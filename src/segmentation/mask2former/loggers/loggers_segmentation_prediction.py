import wandb
import lightning.pytorch as pl
import numpy as np

from src.segmentation.mask2former.utils.utils_logging import get_pixel_mask


class SegmentationPredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, preprocessor=None, img_mean=[0.5, 0.5, 0.5], img_std=[0.15, 0.15, 0.15], num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        self.preprocessor = preprocessor
        self.img_mean = np.array(img_mean)
        self.img_std = np.array(img_std)
        self.val_samples = val_samples

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

    def unnormalize_image(self, normalized_image, means, stds):
        # Ensure the mean and std arrays are broadcastable to the shape of the image
        if means.ndim == 1:
            means = means.reshape(-1, 1, 1)
        if stds.ndim == 1:
            stds = stds.reshape(-1, 1, 1)

        # Unnormalize the image
        original_image = (normalized_image * stds) + means

        return original_image.transpose(1, 2, 0)

    def on_validation_epoch_end(self, trainer, pl_module):
        for sample in self.val_samples:
            imgs, masks, class_labels = sample

            # Bring the tensors to device
            imgs = imgs.to(device=pl_module.device)
            masks = masks.to(device=pl_module.device)
            class_labels = class_labels.to(device=pl_module.device)

            # Get target sized (needed by HuggingFace post processor)
            target_sizes = [(image.shape[1], image.shape[2]) for image in imgs]

            # Infer
            outputs = pl_module(pixel_values=imgs)

            # Get segmentation output
            predicted_segmentation_maps = self.preprocessor.post_process_semantic_segmentation(
                outputs,
                target_sizes=target_sizes
            )

            # Get the prediction segmentation maps into proper valued format
            predicted_segmentation_maps_np = []
            # Iterate over the list and modify each array
            for arr in predicted_segmentation_maps:
                arr = arr.cpu().numpy()
                arr[arr <= 1] = 0
                arr[arr == 2] = 1
                predicted_segmentation_maps_np.append(arr)

            # get class labels
            class_lbls = {
                0: 'background',
                1: 'geographic-atrophy',
            }

            # get pixel masks (b,h,w) from original mask (b,n_classes,h,w)
            pixel_masks_np = get_pixel_mask(masks).cpu().numpy()

            # log prediction and ground truth segmentations for images in batch
            trainer.logger.experiment.log({
                "examples": [
                    wandb.Image(self.unnormalize_image(img.cpu().numpy(), self.img_mean, self.img_std), caption="iou:", masks={
                        "predictions": {
                            "mask_data": pred,
                            "class_labels": class_lbls
                        },
                        "ground_truth": {
                            "mask_data": mask,
                            "class_labels": class_lbls
                        }

                    })
                    for img, mask, pred in zip(imgs, pixel_masks_np, predicted_segmentation_maps_np)
                ]
            })
