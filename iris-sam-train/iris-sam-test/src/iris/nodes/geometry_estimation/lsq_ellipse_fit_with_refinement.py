from typing import List

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

    def run(self, input_polygons: GeometryPolygons) -> GeometryPolygons:
        """Estimate extrapolated polygons with OpenCV's method fitEllipse.

        Args:
            input_polygons (GeometryPolygons): Smoothed polygons.

        Returns:
            GeometryPolygons: Extrapolated polygons.
        """
        extrapolated_pupil = self._extrapolate(input_polygons.pupil_array)
        extrapolated_iris = self._extrapolate(input_polygons.iris_array)

        for point in input_polygons.pupil_array:
            extrapolated_pupil[self._find_correspondence(point, extrapolated_pupil)] = point

        return GeometryPolygons(
            pupil_array=extrapolated_pupil, iris_array=extrapolated_iris, eyeball_array=input_polygons.eyeball_array
        )

    def _extrapolate(self, polygon_points: np.ndarray) -> np.ndarray:
        """Perform extrapolation for points in an array.

        Args:
            polygon_points (np.ndarray): Smoothed polygon ready for applying extrapolation algorithm on it.

        Returns:
            np.ndarray: Estimated extrapolated polygon.
        """
        (x0, y0), (a, b), theta = cv2.fitEllipse(polygon_points)

        extrapolated_polygon = LSQEllipseFitWithRefinement.parametric_ellipsis(
            a / 2, b / 2, x0, y0, np.radians(theta), round(360 / self.params.dphi)
        )

        # Rotate such that 0 degree is parallel with x-axis and array is clockwise
        roll_amount = round((-theta - 90) / self.params.dphi)
        extrapolated_polygon = np.flip(np.roll(extrapolated_polygon, roll_amount, axis=0), axis=0)

        return extrapolated_polygon

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
