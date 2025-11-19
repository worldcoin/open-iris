from typing import List, Union

import cv2
import numpy as np
from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import GeometryPolygons


class LSQEllipseFitWithRefinement(Algorithm):
    """Algorithm that implements least square ellipse fit with iris polygon refinement by finding points to refine by computing euclidean distance.

    Algorithm steps:
        1) Use OpenCV's fitEllipse method to fit an ellipse to predicted iris and pupil polygons.
        2) Refine predicted pupil polygons points to their original location to prevent location precision loss for those points which were predicted by semseg algorithm.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters of least square ellipse fit extrapolation algorithm."""

        dphi: float = Field(..., gt=0.0, lt=360.0)

    __parameters_type__ = Parameters

    def __init__(self, dphi: float = 1.0, callbacks: List[Callback] = []) -> None:
        """Assign parameters.

        Args:
            dphi (float, optional): Angle's delta. Defaults to 1.0.
            callbacks (List[Callback], optional): List of callbacks. Defaults to [].
        """
        super().__init__(dphi=dphi, callbacks=callbacks)

    def run(self, input_polygons: GeometryPolygons) -> Union[GeometryPolygons, None]:
        """Estimate extrapolated polygons with OpenCV's method fitEllipse.

        Args:
            input_polygons (GeometryPolygons): Smoothed polygons.

        Returns:
            Union[GeometryPolygons, None]: Extrapolated polygons or None if pupil is not inside iris.
        """
        extrapolated_polygons = self._extrapolate(input_polygons)

        if extrapolated_polygons is None:
            return None

        for point in input_polygons.pupil_array:
            extrapolated_polygons.pupil_array[
                self._find_correspondence(point, extrapolated_polygons.pupil_array)
            ] = point

        return extrapolated_polygons

    def _is_pupil_inside_iris_ellipses(
        self, px0: float, py0: float, pa: float, pb: float, ix0: float, iy0: float, ia: float, ib: float
    ) -> bool:
        """Fast conservative check using bounding circles.

        Args:
            px0 (float): elliptical fit pupil center x-coordinate
            py0 (float): elliptical fit pupil center y-coordinate
            pa (float): elliptical fit pupil major axis length
            pb (float): elliptical fit pupil minor axis length
            ix0 (float): elliptical fit iris center x-coordinate
            iy0 (float): elliptical fit iris center y-coordinate
            ia (float): elliptical fit iris major axis length
            ib (float): elliptical fit iris minor axis length

        Returns:
            bool: True if pupil is likely inside iris (conservative estimate)
        """
        # Use max radius for pupil (worst case) and min radius for iris (best case)
        pupil_max_radius = max(pa, pb) / 2
        iris_min_radius = min(ia, ib) / 2

        # Distance between centers
        center_dist = np.sqrt((px0 - ix0) ** 2 + (py0 - iy0) ** 2)

        # Conservative check: pupil's max reach < iris's min reach
        # Add small safety margin (e.g., 5%)
        safety_factor = 0.95
        return (center_dist + pupil_max_radius) < (iris_min_radius * safety_factor)

    def _extrapolate(self, input_polygons: GeometryPolygons) -> Union[GeometryPolygons, None]:
        """Perform extrapolation for points in an array.

        Args:
            polygon_points (np.ndarray): Smoothed polygons ready for applying extrapolation algorithm on it.

        Returns:
            Union[GeometryPolygons, None]: Extrapolated polygons or None if pupil is not inside iris.
        """
        (px0, py0), (pa, pb), ptheta = cv2.fitEllipse(input_polygons.pupil_array)
        (ix0, iy0), (ia, ib), itheta = cv2.fitEllipse(input_polygons.iris_array)

        if not self._is_pupil_inside_iris_ellipses(px0, py0, pa, pb, ix0, iy0, ia, ib):
            return None
        extrapolated_pupil_polygon = LSQEllipseFitWithRefinement.parametric_ellipsis(
            pa / 2, pb / 2, px0, py0, np.radians(ptheta), round(360 / self.params.dphi)
        )
        extrapolated_iris_polygon = LSQEllipseFitWithRefinement.parametric_ellipsis(
            ia / 2, ib / 2, ix0, iy0, np.radians(itheta), round(360 / self.params.dphi)
        )

        # Rotate such that 0 degree is parallel with x-axis and array is clockwise
        roll_amount = round((-ptheta - 90) / self.params.dphi)
        extrapolated_pupil_polygon = np.flip(np.roll(extrapolated_pupil_polygon, roll_amount, axis=0), axis=0)
        roll_amount = round((-itheta - 90) / self.params.dphi)
        extrapolated_iris_polygon = np.flip(np.roll(extrapolated_iris_polygon, roll_amount, axis=0), axis=0)

        return GeometryPolygons(
            pupil_array=extrapolated_pupil_polygon,
            iris_array=extrapolated_iris_polygon,
            eyeball_array=input_polygons.eyeball_array,
        )

    def _find_correspondence(self, src_point: np.ndarray, dst_points: np.ndarray) -> int:
        """Find correspondence with Euclidean distance.

        Args:
            src_point (np.ndarray): Source points.
            dst_points (np.ndarray): Destination points.

        Returns:
            int: Source point index the closes one to the destination points.
        """
        src_x, src_y = src_point
        distance = (dst_points[:, 1] - src_y) ** 2 + (dst_points[:, 0] - src_x) ** 2

        idx = np.where(distance == distance.min())[0]
        return idx

    @staticmethod
    def parametric_ellipsis(a: float, b: float, x0: float, y0: float, theta: float, nb_step: int = 100) -> np.ndarray:
        """Given the parameters of a general ellipsis, returns an array of points in this ellipsis.

        Args:
            a (float): Major axis length.
            b (float): Minor axis length.
            x0 (float): x offset.
            y0 (float): y offset.
            theta (float): rotation of the ellipsis.
            nb_step (int): number of points in the ellipsis.

        Returns:
            np.ndarray: points within the ellipsis.
        """
        t = np.linspace(0, 2 * np.pi, nb_step)
        x_coords = x0 + b * np.cos(t) * np.sin(-theta) + a * np.sin(t) * np.cos(-theta)
        y_coords = y0 + b * np.cos(t) * np.cos(-theta) - a * np.sin(t) * np.sin(-theta)

        return np.array([x_coords, y_coords]).T
