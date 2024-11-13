import cv2
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IRImage, NoiseMask


class SpecularReflectionDetection(Algorithm):
    """Apply a threshold to the IR Image to detect specular reflections."""

    class Parameters(Algorithm.Parameters):
        """Parameter class for FusedSemanticSegmentation class."""

        reflection_threshold: int = Field(..., ge=0, le=255)

    __parameters_type__ = Parameters

    def __init__(self, reflection_threshold: int = 254) -> None:
        """Assign parameters.

        Args:
            reflection_threshold (int, optional): Specular Reflection minimal brightness threshold. Defaults to 254.
        """
        super().__init__(reflection_threshold=reflection_threshold)

    def run(self, ir_image: IRImage) -> NoiseMask:
        """Thresholds an IRImage to detect Specular Reflection.

        Args:
            ir_image (IRImage): Infrared image object.

        Returns:
            NoiseMask: a binary map of the thresholded IRImage.
        """
        _, reflection_segmap = cv2.threshold(
            ir_image.img_data, self.params.reflection_threshold, 255, cv2.THRESH_BINARY
        )
        reflection_segmap = (reflection_segmap / 255.0).astype(bool)

        return NoiseMask(mask=reflection_segmap)
