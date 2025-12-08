from __future__ import annotations

import os
from typing import Any, Tuple

import cv2
import numpy as np

from iris.io.class_configs import Algorithm


class MultilabelSemanticSegmentationInterface(Algorithm):
    """Interface of a model semantic segmentation prediction trained with multilabel labels."""

    HUGGING_FACE_REPO_ID = "Worldcoin/iris-semantic-segmentation"
    MODEL_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

    CLASSES_MAPPING = {
        0: "eyeball",
        1: "iris",
        2: "pupil",
        3: "eyelashes",
    }

    @classmethod
    def create_from_hugging_face(cls) -> MultilabelSemanticSegmentationInterface:
        """Abstract function just to make sure all subclasses implement it.

        Raises:
            RuntimeError: Raised if subclass doesn't implement that class method.

        Returns:
            MultilabelSemanticSegmentationInterface: MultilabelSemanticSegmentationInterface subclass object.
        """
        raise RuntimeError(f"`create_from_hugging_face` function hasn't been implemented for {cls.__name__} subclass.")

    def __init__(self, **kwargs: Any) -> None:
        """Assign parameters."""
        super().__init__(**kwargs)

    def image_denoise(
        self, image: np.ndarray, d: int, sigmaColor: int, sigmaSpace: int, intensityIgnore: int
    ) -> np.ndarray:
        """Apply de-noising preprocessing step.

        Args:
            image (np.ndarray): Image to denoise.
            d (int): Diameter of each pixel neighborhood that is used during filtering.
            sigmaColor (int): Filter sigma in color space.
            sigmaSpace (int): Filter sigma in coordinate space.
            intensityIgnore (int): Intensity threshold below which we do not apply filtering.

        Returns:
            np.ndarray: Denoised image.
        """
        # Bilateral filter to reduce noise while preserving edges.
        filtered_image = cv2.bilateralFilter(image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        # Invert the mask to get the areas you do NOT want to filter
        inverse_mask = image < intensityIgnore
        filtered_region = filtered_image * (~inverse_mask)
        # Isolate the original, unfiltered region (where the inverse mask is white)
        original_region = image * inverse_mask
        # Combine the two regions to get the final image
        denoised_image = cv2.add(filtered_region, original_region)

        return denoised_image

    def preprocess(
        self, image: np.ndarray, input_resolution: Tuple[int, int], nn_input_channels: int, denoise: bool
    ) -> np.ndarray:
        """Preprocess image before running a model inference.

        Args:
            image (np.ndarray): Image to preprocess.
            input_resolution (Tuple[int, int]): A model input resolution.
            denoise (bool): Whether to apply de-noising preprocessing step.
            nn_input_channels (int): A model input channels.

        Returns:
            np.ndarray: Preprocessed image.
        """
        nn_input = cv2.resize(image.astype(float), input_resolution)
        if denoise:
            nn_input = self.image_denoise(
                nn_input.astype(np.uint8), d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=75
            )

        nn_input = np.divide(nn_input, 255)  # Replicates torchvision's ToTensor

        nn_input = np.expand_dims(nn_input, axis=-1)
        nn_input = np.tile(nn_input, (1, 1, nn_input_channels))

        # Replicates torchvision's Normalization
        means = np.array([0.485, 0.456, 0.406]) if nn_input_channels == 3 else 0.5
        stds = np.array([0.229, 0.224, 0.225]) if nn_input_channels == 3 else 0.5

        nn_input -= means
        nn_input /= stds

        nn_input = nn_input.transpose(2, 0, 1)
        nn_input = np.expand_dims(nn_input, axis=0)

        return nn_input

    def postprocess_segmap(
        self,
        segmap: np.ndarray,
        original_image_resolution: Tuple[int, int],
    ) -> np.ndarray:
        """Postprocess segmentation map.

        Args:
            segmap (np.ndarray): Predicted segmentation map.
            original_image_resolution (Tuple[int, int]): Original input image resolution (width, height).

        Returns:
            np.ndarray: Postprocessed segmentation map.
        """
        segmap = np.squeeze(segmap, axis=0)
        segmap = np.transpose(segmap, (1, 2, 0))
        segmap = cv2.resize(segmap.astype(np.float32), original_image_resolution, interpolation=cv2.INTER_NEAREST)

        return segmap
