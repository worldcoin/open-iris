from typing import List, Tuple

from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import GeometryMask, NoiseMask, SegmentationMap


class MultilabelSegmentationBinarization(Algorithm):
    """Implementation of a binarization algorithm for multilabel segmentation. Algorithm performs thresholding of each prediction's channel separately to create rasters based on specified by the user classes' thresholds."""

    class Parameters(Algorithm.Parameters):
        """Parameters class for MultilabelSegmentationBinarization objects."""

        eyeball_threshold: float = Field(..., ge=0.0, le=1.0)
        iris_threshold: float = Field(..., ge=0.0, le=1.0)
        pupil_threshold: float = Field(..., ge=0.0, le=1.0)
        eyelashes_threshold: float = Field(..., ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        eyeball_threshold: float = 0.5,
        iris_threshold: float = 0.5,
        pupil_threshold: float = 0.5,
        eyelashes_threshold: float = 0.5,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            eyeball_threshold (float, optional): Eyeball class threshold. Defaults to 0.5.
            iris_threshold (float, optional): Iris class threshold. Defaults to 0.5.
            pupil_threshold (float, optional): Pupil class threshold. Defaults to 0.5.
            eyelashes_threshold (float, optional): Eyelashes class threshold. Defaults to 0.5.
            callbacks (List[Callback], optional): List of algorithm callbacks. Defaults to [].
        """
        super().__init__(
            eyeball_threshold=eyeball_threshold,
            iris_threshold=iris_threshold,
            pupil_threshold=pupil_threshold,
            eyelashes_threshold=eyelashes_threshold,
            callbacks=callbacks,
        )

    def run(self, segmentation_map: SegmentationMap) -> Tuple[GeometryMask, NoiseMask]:
        """Perform segmentation binarization.

        Args:
            segmentation_map (SegmentationMap): Predictions.

        Returns:
            Tuple[GeometryMask, NoiseMask]: Binarized geometry mask and noise mask.
        """
        eyeball_preds = segmentation_map.predictions[..., segmentation_map.index_of("eyeball")]
        iris_preds = segmentation_map.predictions[..., segmentation_map.index_of("iris")]
        pupil_preds = segmentation_map.predictions[..., segmentation_map.index_of("pupil")]
        eyelashes_preds = segmentation_map.predictions[..., segmentation_map.index_of("eyelashes")]

        eyeball_mask = eyeball_preds >= self.params.eyeball_threshold
        iris_mask = iris_preds >= self.params.iris_threshold
        pupil_mask = pupil_preds >= self.params.pupil_threshold
        eyelashes_mask = eyelashes_preds >= self.params.eyelashes_threshold

        return GeometryMask(pupil_mask=pupil_mask, iris_mask=iris_mask, eyeball_mask=eyeball_mask), NoiseMask(
            mask=eyelashes_mask
        )
