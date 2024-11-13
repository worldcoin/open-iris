import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons, IRImage
from iris.io.errors import ExtrapolatedPolygonsInsideImageValidatorError, EyeCentersInsideImageValidatorError


class EyeCentersInsideImageValidator(Algorithm):
    """Validate that the eye center are not too close to the border.

    Raises:
        EyeCentersInsideImageValidatorError: If pupil or iris center are strictly less than `min_distance_to_border`
            pixel of the image boundary.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for EyeCentersInsideImageValidator objects."""

        min_distance_to_border: float

    __parameters_type__ = Parameters

    def __init__(self, min_distance_to_border: float = 0.0) -> None:
        """Assign parameters.

        Args:
            min_distance_to_border (float, optional): Minimum allowed distance to image boundary.
                Defaults to 0.0 (Eye centers can be at the image border).
        """
        super().__init__(min_distance_to_border=min_distance_to_border)

    def run(self, ir_image: IRImage, eye_centers: EyeCenters) -> None:
        """Validate if eye centers are within proper image boundaries.

        Args:
            ir_image (IRImage): IR image
            eye_centers (EyeCenters): Eye centers

        Raises:
            EyeCentersInsideImageValidatorError: Raised if pupil or iris center is not in within correct image boundary.
        """
        if not self._check_center_valid(eye_centers.pupil_x, eye_centers.pupil_y, ir_image):
            raise EyeCentersInsideImageValidatorError("Pupil center is not in allowed image boundary.")

        if not self._check_center_valid(eye_centers.iris_x, eye_centers.iris_y, ir_image):
            raise EyeCentersInsideImageValidatorError("Iris center is not in allowed image boundary.")

    def _check_center_valid(self, center_x: float, center_y: float, ir_image: IRImage) -> bool:
        """Check if center point is within proper image bound.

        Args:
            center_x (float): Center x
            center_y (float): Center y
            ir_image (IRImage): IR image object

        Returns:
            bool: Result of the check.
        """
        return (
            self.params.min_distance_to_border <= center_x <= ir_image.width - self.params.min_distance_to_border
            and self.params.min_distance_to_border <= center_y <= ir_image.height - self.params.min_distance_to_border
        )


class ExtrapolatedPolygonsInsideImageValidator(Algorithm):
    """Validate that GeometryPolygons are included within the image to a certain minimum percentage.

    Raises:
        ExtrapolatedPolygonsInsideImageValidatorError: If the number of points of the pupil/iris/eyeball
            that are within the input image is below threshold.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for ExtrapolatedPolygonsInsideImageValidator objects."""

        min_pupil_allowed_percentage: float = Field(..., ge=0.0, le=1.0)
        min_iris_allowed_percentage: float = Field(..., ge=0.0, le=1.0)
        min_eyeball_allowed_percentage: float = Field(..., ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        min_pupil_allowed_percentage: float = 0.0,
        min_iris_allowed_percentage: float = 0.0,
        min_eyeball_allowed_percentage: float = 0.0,
    ) -> None:
        """Assign parameters.

        Args:
            min_pupil_allowed_percentage (float, optional): Minimum allowed percentage of extrapolated pupil polygons that must be within an image.
                Defaults to 0.0 (Entire extrapolated polygon may be outside of an image).
            min_iris_allowed_percentage (float, optional): Minimum allowed percentage of extrapolated iris polygons that must be within an image.
                Defaults to 0.0 (Entire extrapolated polygon may be outside of an image).
            min_eyeball_allowed_percentage (float, optional): Minimum allowed percentage of extrapolated eyeball polygons that must be within an image.
                Defaults to 0.0 (Entire extrapolated polygon may be outside of an image).
        """
        super().__init__(
            min_pupil_allowed_percentage=min_pupil_allowed_percentage,
            min_iris_allowed_percentage=min_iris_allowed_percentage,
            min_eyeball_allowed_percentage=min_eyeball_allowed_percentage,
        )

    def run(self, ir_image: IRImage, extrapolated_polygons: GeometryPolygons) -> None:
        """Perform validation.

        Args:
            ir_image (IRImage): IR image.
            extrapolated_polygons (GeometryPolygons): Extrapolated polygons.

        Raises:
            ExtrapolatedPolygonsInsideImageValidatorError: Raised if not enough points of the pupil/iris/eyeball are within an image.
        """
        if not self._check_correct_percentage(
            extrapolated_polygons.pupil_array, self.params.min_pupil_allowed_percentage, ir_image
        ):
            raise ExtrapolatedPolygonsInsideImageValidatorError("Not enough pupil points are within an image.")

        if not self._check_correct_percentage(
            extrapolated_polygons.iris_array, self.params.min_iris_allowed_percentage, ir_image
        ):
            raise ExtrapolatedPolygonsInsideImageValidatorError("Not enough iris points are within an image.")

        if not self._check_correct_percentage(
            extrapolated_polygons.eyeball_array, self.params.min_eyeball_allowed_percentage, ir_image
        ):
            raise ExtrapolatedPolygonsInsideImageValidatorError("Not enough eyeball points are within an image.")

    def _check_correct_percentage(self, polygon: np.ndarray, min_allowed_percentage: float, ir_image: IRImage) -> bool:
        """Check percentage of points withing image based on minimal specified threshold.

        Args:
            polygon (np.ndarray): polygon to verify.
            min_allowed_percentage (float): minimal allowed percentage of points that must be within an image.
            ir_image (IRImage): ir image object.

        Returns:
            bool: Check result.
        """
        num_points_inside_image: float = np.sum(
            np.all(np.logical_and((0, 0) <= polygon, polygon <= (ir_image.width, ir_image.height)), axis=1)
        )

        percentage_points_inside_image = num_points_inside_image / len(polygon)

        return percentage_points_inside_image >= min_allowed_percentage
