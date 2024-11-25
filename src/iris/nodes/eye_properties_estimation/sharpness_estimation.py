from typing import List, Tuple

import cv2
import numpy as np
from pydantic import Field, validator

import iris.io.validators as pydantic_v
from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import NormalizedIris, Sharpness


class SharpnessEstimation(Algorithm):
    """Calculate sharpness of the normalized iris.

    The goal of this algorithm is to calculate the sharpness of the normalized iris using the variance of Laplacian.

    LIMITATIONS:

    This method may be biased against dark images with inadequate lighting.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for SharpnessEstimation objects.

        lap_ksize (int): Laplacian kernel size, must be odd integer no larger than 31.
        erosion_ksize (Tuple[int, int]): Mask erosion kernel size, must be odd integers.
        """

        lap_ksize: int = Field(..., gt=0, le=31)
        erosion_ksize: Tuple[int, int] = Field(..., gt=0)

        _is_odd0 = validator("lap_ksize", allow_reuse=True)(pydantic_v.is_odd)
        _is_odd = validator("erosion_ksize", allow_reuse=True, each_item=True)(pydantic_v.is_odd)

    __parameters_type__ = Parameters

    def __init__(
        self,
        lap_ksize: int = 11,
        erosion_ksize: Tuple[int, int] = (29, 15),
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            lap_ksize (int, optional): kernal size for Laplacian. Defaults to 11.
            erosion_ksize (Tuple[int, int], optional): kernal size for mask erosion. Defaults to (29,15).
            callbacks (List[Callback]): callbacks list. Defaults to [].
        """
        super().__init__(lap_ksize=lap_ksize, erosion_ksize=erosion_ksize, callbacks=callbacks)

    def run(self, normalization_output: NormalizedIris) -> Sharpness:
        """Calculate sharpness of the normalized iris.

        Args:
            normalization_output (NormalizedIris): Normalized iris.

        Returns:
            Sharpness: Sharpness object.
        """
        output_im = cv2.Laplacian(normalization_output.normalized_image / 255, cv2.CV_32F, ksize=self.params.lap_ksize)
        mask_im = cv2.erode(
            normalization_output.normalized_mask.astype(np.uint8), kernel=np.ones(self.params.erosion_ksize, np.uint8)
        )
        sharpness_score = output_im[mask_im == 1].std() if np.sum(mask_im == 1) > 0 else 0.0
        return Sharpness(score=sharpness_score)
