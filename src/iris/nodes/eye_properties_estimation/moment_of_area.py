import cv2
from pydantic import Field

import iris.utils.math as math_utils
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeOrientation, GeometryPolygons
from iris.io.errors import EyeOrientationEstimationError


class MomentOfArea(Algorithm):
    """Estimate the eye orientation using the second order moments of the eyeball polygon.

    The eye orientation refers to the horizontal direction of the eye. It comes useful for determining the
    partial eye occlusion (e.g. occlusion at the horizontal middle third of the iris).

    References:
        [1] https://t1.daumcdn.net/cfile/tistory/15425F4150F4EBFC19
        [2] https://en.wikipedia.org/wiki/Image_moment
    """

    class Parameters(Algorithm.Parameters):
        """MomentOfArea parameters.

        eccentricity_threshold: float in [0, 1].
            The threshold below which a shape is considered not linear enough to reliably estimate its orientation.
        """

        eccentricity_threshold: float = Field(ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(self, eccentricity_threshold: float = 0.1) -> None:
        """Assign parameters.

        Args:
            eccentricity_threshold: float in [0, 1]. The threshold below which a shape is considered not linear enough to reliably estimate its orientation. Defaults to 0.1.
        """
        super().__init__(eccentricity_threshold=eccentricity_threshold)

    def run(self, geometries: GeometryPolygons) -> EyeOrientation:
        """Compute the eye orientation using the second order moments or the eyeball.

        WARNING: cv2.moments MUST only receive np.float32 arrays. Otherwise, the array will be interpreted as a sparse
        matrix instead of a list of points. See https://github.com/opencv/opencv/issues/6643#issuecomment-224204774.

        Args:
            geometries (GeometryPolygons): segmentation map used for eye orientation estimation.

        Raises:
            EyeOrientationEstimationError if the eyeball's eccentricity is below `eccentricity_threshold` i.e. if the eyeball shape is not circular enough to reliably estimate the orientation.

        Returns:
            EyeOrientation: eye orientation object.
        """
        moments = cv2.moments(geometries.eyeball_array)

        eccentricity = math_utils.eccentricity(moments)
        if eccentricity < self.params.eccentricity_threshold:
            raise EyeOrientationEstimationError(
                "The eyeball is too circular to reliably determine its orientation. "
                f"Computed eccentricity: {eccentricity}. Threshold: {self.params.eccentricity_threshold}"
            )

        orientation = math_utils.orientation(moments)
        return EyeOrientation(angle=orientation)
