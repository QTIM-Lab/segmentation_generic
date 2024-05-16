import numpy as np


def get_pixel_mask(label_mask, class_offset=0):
    '''
    Class offset sometimes needed if you want to have it be 1 = bg, 2 = ga, or sometimes 0 = bg, 1 = ga, etc.

    Takes a (batch, n_classes, height, width) shaped array and returns a (batch, height, width) with proper values
    '''
    # Create a new NumPy array with the desired shape (batch_size, height, width)
    new_mask = np.zeros((label_mask.shape[0], label_mask.shape[2], label_mask.shape[3]), dtype=np.uint8)

    # Iterate over the batch dimension
    for batch in range(label_mask.shape[0]):
        # Set the pixel values based on the channel values
        new_mask[batch][label_mask[batch, 0] == 1] = 0 + class_offset
        new_mask[batch][label_mask[batch, 1] == 1] = 1 + class_offset

    return new_mask