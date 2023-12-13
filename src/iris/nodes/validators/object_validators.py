from typing import Tuple

import numpy as np
from pydantic import Field

import iris.io.errors as E
from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeOcclusion, GeometryPolygons, IrisTemplate, Offgaze, PupilToIrisProperty
from iris.utils.math import polygon_length


class Pupil2IrisPropertyValidator(Callback, Algorithm):
    """Validate that the pupil-to-iris ratio is within thresholds.

    Raises:
        E.PupilIrisPropertyEstimationError: If the pupil-to-iris ratio isn't within boundaries.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for Pupil2IrisPropertyValidator objects."""

        min_allowed_diameter_ratio: float = Field(..., gt=0.0, lt=1.0)
        max_allowed_diameter_ratio: float = Field(..., gt=0.0, lt=1.0)
        max_allowed_center_dist_ratio: float = Field(..., ge=0.0, lt=1.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        min_allowed_diameter_ratio: float = 0.0001,
        max_allowed_diameter_ratio: float = 0.9999,
        max_allowed_center_dist_ratio: float = 0.9999,
    ) -> None:
        """Assign parameters.

        Args:
            min_allowed_diameter_ratio (float): Minimum allowed pupil2iris diameter ratio. Defaults to 0.0001 (by default every check will result in success).
            max_allowed_diameter_ratio (float): Maximum allowed pupil2iris diameter ratio. Defaults to 0.9999 (by default every check will result in success).
            max_allowed_center_dist_ratio (float): Maximum allowed pupil2iris center distance ratio. Defaults to 0.9999 (by default every check will result in success).
        """
        super().__init__(
            min_allowed_diameter_ratio=min_allowed_diameter_ratio,
            max_allowed_diameter_ratio=max_allowed_diameter_ratio,
            max_allowed_center_dist_ratio=max_allowed_center_dist_ratio,
        )

    def run(self, val_arguments: PupilToIrisProperty) -> None:
        """Validate of pupil to iris calculation.

        Args:
            p2i_property (PupilToIrisProperty): Computation result.

        Raises:
            E.PupilIrisPropertyEstimationError: Raised if result isn't without previously specified boundaries.
        """
        if not (
            self.params.min_allowed_diameter_ratio
            <= val_arguments.pupil_to_iris_diameter_ratio
            <= self.params.max_allowed_diameter_ratio
        ):
            raise E.PupilIrisPropertyEstimationError(
                f"p2i_property={val_arguments.pupil_to_iris_diameter_ratio} is not within [{self.params.min_allowed_diameter_ratio}, {self.params.max_allowed_diameter_ratio}]."
            )
        if val_arguments.pupil_to_iris_center_dist_ratio > self.params.max_allowed_center_dist_ratio:
            raise E.PupilIrisPropertyEstimationError(
                f"p2i_property={val_arguments.pupil_to_iris_center_dist_ratio} exceeds {self.params.max_allowed_center_dist_ratio}."
            )

    def on_execute_end(self, result: PupilToIrisProperty) -> None:
        """Wrap for validate method so that validator can be used as a Callback.

        Args:
            result (PupilToIrisProperty): Pupil2Iris property resulted from computations.
        """
        self.run(result)


class OffgazeValidator(Callback, Algorithm):
    """Validate that the offgaze score is below threshold.

    Raises:
        E.OffgazeEstimationError: If the offgaze score is above threshold.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for OffgazeValidator objects."""

        max_allowed_offgaze: float = Field(..., ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(self, max_allowed_offgaze: float = 1.0) -> None:
        """Assign parameters.

        Args:
            max_allowed_offgaze (float): Offgaze computation result max threshold that allows further sample processing.
                Defaults to 1.0 (by default every check will result in success).
        """
        super().__init__(max_allowed_offgaze=max_allowed_offgaze)

    def run(self, val_arguments: Offgaze) -> None:
        """Validate of offgaze estimation algorithm.

        Args:
            val_arguments (Offgaze): Computed result.

        Raises:
            E.OffgazeEstimationError: Raised if result isn't greater then specified threshold.
        """
        if not (val_arguments.score <= self.params.max_allowed_offgaze):
            raise E.OffgazeEstimationError(
                f"offgaze={val_arguments.score} > max_allowed_offgaze={self.params.max_allowed_offgaze}"
            )

    def on_execute_end(self, result: Offgaze) -> None:
        """Wrap for validate method so that validator can be used as a Callback.

        Args:
            result (Offgaze): Offgaze resulted from computations.
        """
        self.run(result)


class OcclusionValidator(Callback, Algorithm):
    """Validate that the occlusion fration is above threshold.

    Raises:
        E.OcclusionError: If the occlusion fraction is below threshold.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for OcclusionValidator objects."""

        min_allowed_occlusion: float = Field(..., ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(self, min_allowed_occlusion: float = 0.0) -> None:
        """Assign parameters.

        Args:
            min_allowed_occlusion (float): Occlusion computation result min threshold that allows further sample processing.
                Defaults to 0.0 (by default every check will result in success).
        """
        super().__init__(min_allowed_occlusion=min_allowed_occlusion)

    def run(self, val_arguments: EyeOcclusion) -> None:
        """Validate of occlusion estimation algorithm.

        Args:
            val_arguments (EyeOcclusion): Computed result.

        Raises:
            E.OcclusionError: Raised if result isn't greater then specified threshold.
        """
        if not (val_arguments.visible_fraction >= self.params.min_allowed_occlusion):
            raise E.OcclusionError(
                f"visible_fraction={val_arguments.visible_fraction} < min_allowed_occlusion={self.params.min_allowed_occlusion}."
            )

    def on_execute_end(self, result: EyeOcclusion) -> None:
        """Wrap for validate method so that validator can be used as a Callback.

        Args:
            result (EyeOcclusion): EyeOcclusion resulted from computations.
        """
        self.run(result)


class IsPupilInsideIrisValidator(Callback, Algorithm):
    """Validate that the pupil is fully contained within the iris.

    Raises:
        E.IsPupilInsideIrisValidatorError: If the pupil polygon is not fully contained within the iris polygon.
    """

    def run(self, val_arguments: GeometryPolygons) -> None:
        """Validate if extrapolated pupil polygons are withing extrapolated iris boundaries.

        Args:
            val_arguments (GeometryPolygons): Computed result.

        Raises:
            E.IsPupilInsideIrisValidatorError: Raised if the pupil polygon is not fully contained within the iris polygon.
        """
        for point in val_arguments.pupil_array:
            if not self._check_pupil_point_is_inside_iris(point, val_arguments.iris_array):
                raise E.IsPupilInsideIrisValidatorError(
                    "Entire extrapolated pupil polygon isn't included in an extrapolated iris polygon."
                )

    def on_execute_end(self, result: GeometryPolygons) -> None:
        """Wrap for validate method so that validator can be used as a Callback.

        Args:
            result (GeometryPolygons): GeometryPolygons resulted from computations.
        """
        self.run(result)

    def _check_pupil_point_is_inside_iris(self, point: np.ndarray, polygon_pts: np.ndarray) -> bool:
        """Check if pupil point is inside iris polygon.

        Reference:
            [1] https://www.geeksforgeeks.org/how-to-check-if-a-given-point-lies-inside-a-polygon/

        Args:
            point (np.ndarray): Point x, y.
            polygon_sides (np.ndarray): Polygon points.

        Returns:
            bool: Check result.
        """
        num_iris_points = len(polygon_pts)
        polygon_sides = [
            (polygon_pts[i % num_iris_points], polygon_pts[(i + 1) % num_iris_points]) for i in range(num_iris_points)
        ]

        x, y = point
        to_right_ray = (point, np.array([float("inf"), y]))
        to_left_ray = (np.array([-float("inf"), y]), point)

        right_ray_intersections, left_ray_intersections = 0, 0
        for poly_side in polygon_sides:
            if self._is_ray_intersecting_with_side(to_right_ray, poly_side, is_ray_pointing_to_left=False):
                right_ray_intersections += 1

            if self._is_ray_intersecting_with_side(to_left_ray, poly_side, is_ray_pointing_to_left=True):
                left_ray_intersections += 1

        return right_ray_intersections % 2 != 0 or left_ray_intersections % 2 != 0

    def _is_ray_intersecting_with_side(
        self,
        ray_line: Tuple[np.ndarray, np.ndarray],
        side_line: Tuple[np.ndarray, np.ndarray],
        is_ray_pointing_to_left: bool,
    ) -> bool:
        """Check if ray is intersecting with a polygon side.

        Args:
            ray_line (Tuple[np.ndarray, np.ndarray]): Ray line two points.
            side_line (Tuple[np.ndarray, np.ndarray]): Side line two points.
            is_ray_pointing_to_left (bool): Is ray pointing to the left flag.

        Returns:
            bool: Check result.
        """
        (ray_start_x, ray_start_y), (ray_end_x, ray_end_y) = ray_line
        (side_start_x, side_start_y), (side_end_x, side_end_y) = side_line

        if side_start_y == side_end_y:
            return side_start_y == ray_start_y

        # fmt: off
        intersection_x = (ray_start_y - side_start_y) * (side_start_x - side_end_x) / (side_start_y - side_end_y) + side_start_x
        # fmt: on

        is_along_side = side_start_x <= intersection_x < side_end_x or side_start_x >= intersection_x > side_end_x
        is_along_ray = intersection_x <= ray_end_x if is_ray_pointing_to_left else intersection_x >= ray_start_x

        return is_along_side and is_along_ray


class PolygonsLengthValidator(Callback, Algorithm):
    """Validate that the pupil and iris polygons have a sufficient length.

    Raises:
        E.GeometryEstimationError: If the total iris or pupil polygon length is below the desired threshold.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for PolygonsLengthValidator objects."""

        min_iris_length: int = Field(..., ge=0)
        min_pupil_length: int = Field(..., ge=0)

    __parameters_type__ = Parameters

    def __init__(self, min_iris_length: int = 150, min_pupil_length: int = 75) -> None:
        """Assign parameters.

        Args:
            min_iris_length (int): Minimum cumulated length of the iris polygon. If too small, the extrapolation algorithm won't work properly. Defaults to 150.
            min_pupil_length (int): Minimum cumulated length of the pupil polygon. If too small, the extrapolation algorithm won't work properly. Defaults to 75.
        """
        super().__init__(min_iris_length=min_iris_length, min_pupil_length=min_pupil_length)

    def run(self, val_arguments: GeometryPolygons) -> None:
        """Validate that the total iris and pupil polygon length is above the desired threshold.

        Args:
            val_arguments (GeometryPolygons): GeometryPolygons to be validated.

        Raises:
            E.GeometryEstimationError: Raised if the total iris or pupil polygon length is below the desired threshold.
        """
        pupil_length = polygon_length(val_arguments.pupil_array)
        iris_length = polygon_length(val_arguments.iris_array)

        if pupil_length < self.params.min_pupil_length:
            raise E.GeometryEstimationError(
                f"Valid pupil polygon is too small: Got {pupil_length} px, min {self.params.min_pupil_length} px."
            )
        if iris_length < self.params.min_iris_length:
            raise E.GeometryEstimationError(
                f"Valid iris polygon is too small: Got {iris_length} px, min {self.params.min_iris_length} px."
            )

    def on_execute_start(self, input_polygons: GeometryPolygons, *args, **kwargs) -> None:
        """Wrap for validate method so that validator can be used as a Callback.

        Args:
            input_polygons (GeometryPolygons): input GeometryPolygons to be validated.
        """
        self.run(input_polygons)


class IsMaskTooSmallValidator(Callback, Algorithm):
    """Validate that the masked part of the IrisTemplate is small enough.

    The larger the mask, the less reliable information is available to create a robust identity.

    Raises:
        E.EncoderError: If the total number of non-masked bits is below threshold.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for IsMaskTooSmallValidator objects."""

        min_maskcodes_size: int = Field(..., ge=0)

    __parameters_type__ = Parameters

    def __init__(self, min_maskcodes_size: int = 0) -> None:
        """Assign parameters.

        Args:
            min_maskcodes_size (int): Minimum size of mask codes. If too small, valid iris texture is too small, should be rejected.
        """
        super().__init__(min_maskcodes_size=min_maskcodes_size)

    def run(self, val_arguments: IrisTemplate) -> None:
        """Validate that the total mask codes size is above the desired threshold.

        Args:
            val_arguments (IrisTemplate): IrisTemplate to be validated.

        Raises:
            E.EncoderError: Raised if the total mask codes size is below the desired threshold.
        """
        maskcodes_size = np.sum(val_arguments.mask_codes)

        if maskcodes_size < self.params.min_maskcodes_size:
            raise E.EncoderError(
                f"Valid mask codes size is too small: Got {maskcodes_size} px, min {self.params.min_maskcodes_size} px."
            )

    def on_execute_end(self, input_template: IrisTemplate, *args, **kwargs) -> None:
        """Wrap for validate method so that validator can be used as a Callback.

        Args:
            input_template (IrisTemplate): input IrisTemplate to be validated.
        """
        self.run(input_template)
