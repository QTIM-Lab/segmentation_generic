import os
from PIL import Image
import numpy as np

from src.segmentation.generic.data_modules.datasets_generic import GenericDataset


class Mask2FormerDataset(GenericDataset):
    """
    Mask2Former dataset.
    """

    def __init__(self, images, masks, transform, preprocess, image_root_dir, label_root_dir):
        super().__init__(images, masks, transform, preprocess, image_root_dir, label_root_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Get Image
        path_to_img = os.path.join(self.image_root_dir, self.images[idx])
        original_image = np.array(Image.open(path_to_img))

        original_image_width = original_image.shape[0]
        original_image_height = original_image.shape[1]

        # Get mask if available. Otherwise, mask can be just all 0's
        if self.masks is None or self.label_root_dir is None:
            # Create a new array with the same shape, filled with ones (background)
            original_segmentation_map = np.ones((original_image_width, original_image_height), dtype=np.uint8)
            original_segmentation_map[0, 0] = 2
        else:
            path_to_mask = os.path.join(self.mask_root_dir, self.masks[idx])
            original_segmentation_map = np.array(Image.open(path_to_mask).convert('L'))
            # TODO: Fix hardcode
            original_segmentation_map[original_segmentation_map == 0] = 1
            original_segmentation_map[original_segmentation_map == 255] = 2

        # Get transformed image
        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, segmentation_map = transformed['image'], transformed['mask']

        # convert to C, H, W
        image = image.transpose(2, 0, 1)

        # Get data in way that HuggingFace Mask2Former wants
        batched = self.preprocessor(
            image,
            segmentation_maps=segmentation_map,
            return_tensors="pt",
        )
        # Note all had batch dim, even though just one item, so take 0 index to get rid of that here
        pixel_values = batched['pixel_values'][0]  # (3, 512, 512)
        mask_labels = batched['mask_labels'][0]  # (2, 512, 512)
        class_labels = batched['class_labels'][0]  # (2,)

        return pixel_values, mask_labels, class_labels, (original_image_width, original_image_height), self.images[idx]
