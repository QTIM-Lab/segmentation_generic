import numpy as np
from torch.utils.data import Dataset
from typing import Tuple, Any
import torch


class GenericDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, np.ndarray[Any, Any], np.ndarray[Any, Any]]]):
    """
    Mask2Former dataset.
    """

    def __init__(self, images, masks, transform, preprocess, image_root_dir, label_root_dir):
        """
        Args:
            dataset
        """
        self.images = images
        self.masks = masks
        self.transform = transform
        self.image_root_dir = image_root_dir
        self.preprocessor = preprocess
        self.mask_root_dir = label_root_dir

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return idx
