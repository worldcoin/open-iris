from typing import List

import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import GeometryPolygons


class ContourInterpolation(Algorithm):
    """Implementation of contour interpolation algorithm conditioned by given NoiseMask.

    Algorithm performs linar interpolation of points between vectorized, predicted points such that maximum distance between two consecutive points in a polygon isn't greater than
    a fraction of an iris diameter length specified as `max_distance_between_boundary_points` parameter.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for ContourInterpolation objects."""

        max_distance_between_boundary_points: float = Field(..., gt=0.0, lt=1.0)

    __parameters_type__ = Parameters

    def __init__(self, max_distance_between_boundary_points: float = 0.01) -> None:
        """Assign parameters.

        Args:
            max_distance_between_boundary_points (float, optional): Maximum distance between boundary contour points expressed as a fraction of a iris diameter length. Defaults to 0.01.
        """
        super().__init__(max_distance_between_boundary_points=max_distance_between_boundary_points)

    def run(self, polygons: GeometryPolygons) -> GeometryPolygons:
        """Refine polygons by interpolating contour points.

        Args:
            polygons (GeometryPolygons): Polygons to refine.

        Returns:
            GeometryPolygons: Refined polygons.
        """
        max_boundary_dist_in_px = self.params.max_distance_between_boundary_points * polygons.iris_diameter

        refined_pupil_array = self._interpolate_polygon_points(polygons.pupil_array, max_boundary_dist_in_px)
        refined_iris_array = self._interpolate_polygon_points(polygons.iris_array, max_boundary_dist_in_px)
        refined_eyeball_array = self._interpolate_polygon_points(polygons.eyeball_array, max_boundary_dist_in_px)

        return GeometryPolygons(
            pupil_array=refined_pupil_array,
            iris_array=refined_iris_array,
            eyeball_array=refined_eyeball_array,
        )

    def _interpolate_polygon_points(self, polygon: np.ndarray, max_distance_between_points_px: float) -> np.ndarray:
        """Interpolate contours points, so that the distance between two is no greater than `self.params.max_distance_between_boundary_points` in pixel space.

        Args:
            polygon (np.ndarray): Contour polygons.
            max_distance_between_points_px (float): `self.params.max_distance_between_boundary_points` expressed in pixel length relative to iris diameter.

        Returns:
            np.ndarray: Interpolated polygon points.
        """
        previous_boundary = np.roll(polygon, shift=1, axis=0)
        distances = np.linalg.norm(polygon - previous_boundary, axis=1)
        num_points = np.ceil(distances / max_distance_between_points_px).astype(int)

        x: List[np.ndarray] = []
        y: List[np.ndarray] = []
        for (x1, y1), (x2, y2), num_point in zip(previous_boundary, polygon, num_points):
            x.append(np.linspace(x1, x2, num=num_point, endpoint=False))
            y.append(np.linspace(y1, y2, num=num_point, endpoint=False))

        new_boundary = np.stack([np.concatenate(x), np.concatenate(y)], axis=1)
        _, indices = np.unique(new_boundary, axis=0, return_index=True)
        new_boundary = new_boundary[np.sort(indices)]

        return new_boundary
