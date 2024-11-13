from typing import Tuple

import cv2
import numpy as np


def contour_to_mask(vertices: np.ndarray, mask_shape: Tuple[int, int]) -> np.ndarray:
    """Generate binary mask based on polygon's vertices.

    Args:
        vertices (np.ndarray): Vertices points array.
        mask_shape (Tuple[int, int]): Tuple with output mask dimension (weight, height).

    Returns:
        np.ndarray: Binary mask.
    """
    width, height = mask_shape
    mask = np.zeros(shape=(height, width, 3))

    vertices = np.round(vertices).astype(np.int32)
    cv2.fillPoly(mask, pts=[vertices], color=(255, 0, 0))

    mask = mask[..., 0]
    mask = mask.astype(bool)

    return mask
