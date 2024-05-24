import numpy as np
import random


def get_label_bbox(gt, bbox_shift=20):
    y_indices, x_indices = np.where(gt > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    h, w = gt.shape
    zero_a, zero_b = np.array([0, 0])
    x_min = max(zero_a, x_min - random.randint(0, bbox_shift))
    x_max = min(h, x_max + random.randint(0, bbox_shift))
    y_min = max(zero_b, y_min - random.randint(0, bbox_shift))
    y_max = min(w, y_max + random.randint(0, bbox_shift))
    bboxes = [float(x_min), float(y_min), float(x_max), float(y_max)]
    return bboxes
