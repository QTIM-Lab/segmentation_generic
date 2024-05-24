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
        # Get paths
        path_to_img = os.path.join(self.image_root_dir, self.images[idx])
        path_to_mask = os.path.join(self.mask_root_dir, self.masks[idx])

        # Open image
        original_image = np.array(Image.open(path_to_img))
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

        return pixel_values, mask_labels, class_labels
