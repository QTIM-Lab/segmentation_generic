import os
from PIL import Image
import numpy as np

from src.segmentation.generic.data_modules.datasets_generic import GenericDataset
from src.segmentation.medsam.utils.utils_medsam import get_label_bbox


class MedSAMDataset(GenericDataset):
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
        original_image = np.array(Image.open(path_to_img).convert('RGB'))
        original_segmentation_map = np.array(Image.open(path_to_mask).convert('L'))
        # TODO: Fix hardcode
        original_segmentation_map[original_segmentation_map > 0] = 1

        # Get transformed image
        transformed = self.transform(image=original_image, mask=original_segmentation_map)
        image, mask_label = transformed['image'], transformed['mask']

        # Convert image to C, H, W
        image = image.transpose(2, 0, 1)

        # Get the bounding boxes of the labels
        input_boxes = [get_label_bbox(mask_label)]

        # Get data in way that MedSAM wants
        batched = self.preprocessor(
            [image],
            input_boxes=[[input_boxes]],
            return_tensors="pt",
            do_rescale=False
        )

        # Note all had batch dim, even though just one item, so take 0 index to get rid of that here
        pixel_values = batched['pixel_values'][0]  # (3, 1024, 1024)
        original_sizes = batched['original_sizes'][0]  # (2,) original height and width
        reshaped_input_sizes = batched['reshaped_input_sizes'][0]  # (2,) resized height and width
        input_boxes = batched['input_boxes'][0]  # (4,) [x1,y1,x2,y2]

        return pixel_values, mask_label, original_sizes, reshaped_input_sizes, input_boxes
