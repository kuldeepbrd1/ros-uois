from typing import Union

import numpy as np
import torch


def alpha_blend(rgb_image, segmentation_mask, alpha=0.5):
    # Ensure the mask and image have the same dimensions
    assert (
        rgb_image.shape == segmentation_mask.shape
    ), "Image and mask must have the same dimensions."

    # Create a new array for the blended result
    blended_image = np.copy(rgb_image)

    # Apply alpha blending by combining the image and the mask
    blended_image = (alpha * rgb_image + (1 - alpha) * segmentation_mask).astype(
        np.uint8
    )

    return blended_image
