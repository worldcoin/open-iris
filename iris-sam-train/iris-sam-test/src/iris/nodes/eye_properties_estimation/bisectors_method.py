from typing import Tuple

import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.io.errors import EyeCentersEstimationError


class BisectorsMethod(Algorithm):
    """Implementation of eye's center estimation algorithm using bisectors method for finding a circle center.

    This algorithm samples a given number of bisectors from the pupil and iris polygons, and averages their intersection
    to produce the polygon center. This method is robust against noise in the polygons, making it a good choice for
    non-perfect shapes. It is also robust to polygons missing parts of the circle arc, making it a good choice for
    partially-occluded shapes.

    LIMITATIONS:
    The iris and pupil can be approximated to circles, when the user is properly gazing at the camera.
    This requires that the cases of off-gaze have already been filtered out.
    """

    class Parameters(Algorithm.Parameters):
        """Default Parameters for BisectorsMethod algorithm."""

        num_bisectors: int = Field(..., gt=0)
        min_distance_between_sector_points: float = Field(..., gt=0.0, lt=1.0)
        max_iterations: int = Field(..., gt=0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        num_bisectors: int = 100,
        min_distance_between_sector_points: float = 0.75,
        max_iterations: int = 50,
    ) -> None:
        """Assign parameters.

        Args:
            num_bisectors (int, optional): Number of bisectors.. Defaults to 100.
            min_distance_between_sector_points (float, optional): Minimum distance between sectors expressed as a fractional value of a circular shape diameter. Defaults to 0.75.
            max_iterations (int, optional): Max iterations for bisector search.. Defaults to 50.
        """
        super().__init__(
            num_bisectors=num_bisectors,
            min_distance_between_sector_points=min_distance_between_sector_points,
            max_iterations=max_iterations,
        )

    def run(self, geometries: GeometryPolygons) -> EyeCenters:
        """Estimate eye's iris and pupil centers.

        Args:
            geometries (GeometryPolygons): Geometry polygons.

        Returns:
            EyeCenters: Eye's centers object.
        """
        pupil_center_x, pupil_center_y = self._find_center_coords(geometries.pupil_array, geometries.pupil_diameter)
        iris_center_x, iris_center_y = self._find_center_coords(geometries.iris_array, geometries.iris_diameter)

        return EyeCenters(pupil_x=pupil_center_x, pupil_y=pupil_center_y, iris_x=iris_center_x, iris_y=iris_center_y)

    def _find_center_coords(self, polygon: np.ndarray, diameter: float) -> Tuple[float, float]:
        """Find center coordinates of a polygon.

        Args:
            polygon (np.ndarray): np.ndarray.
            diameter (float): diameter of the polygon.

        Returns:
            Tuple[float, float]: Tuple with the center location coordinates (x, y).
        """
        min_distance_between_sector_points_in_px = self.params.min_distance_between_sector_points * diameter

        first_bisectors_point, second_bisectors_point = self._calculate_perpendicular_bisectors(
            polygon, min_distance_between_sector_points_in_px
        )

        return self._find_best_intersection(first_bisectors_point, second_bisectors_point)

    def _calculate_perpendicular_bisectors(
        self, polygon: np.ndarray, min_distance_between_sector_points_in_px: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate the perpendicular bisector of self.params.num_bisectors randomly chosen points from a polygon's vertices.
            A pair of points is used if their distance is larger then min_distance_between_sector_points_in_px.

        Args:
            polygon (np.ndarray): np.ndarray based on which we are searching the center of a circular shape.
            min_distance_between_sector_points_in_px (float): Minimum distance between sector points.

        Raises:
            EyeCentersEstimationError: Raised if not able to find enough random pairs of points on the arc with a large enough distance!

        Returns:
            Tuple[np.ndarray, np.ndarray]: Calculated perpendicular bisectors.
        """
        np.random.seed(142857)

        bisectors_first_points = np.empty([0, 2])
        bisectors_second_points = np.empty([0, 2])
        for _ in range(self.params.max_iterations):
            random_indices = np.random.choice(len(polygon), size=(self.params.num_bisectors, 2))

            first_drawn_points = polygon[random_indices[:, 0]]
            second_drawn_points = polygon[random_indices[:, 1]]

            norms = np.linalg.norm(first_drawn_points - second_drawn_points, axis=1)
            mask = norms > min_distance_between_sector_points_in_px

            bisectors_first_points = np.vstack([bisectors_first_points, first_drawn_points[mask]])
            bisectors_second_points = np.vstack([bisectors_second_points, second_drawn_points[mask]])

            if len(bisectors_first_points) >= self.params.num_bisectors:
                break
        else:
            raise EyeCentersEstimationError(
                "Not able to find enough random pairs of points on the arc with a large enough distance!"
            )

        bisectors_first_points = bisectors_first_points[: self.params.num_bisectors]
        bisectors_second_points = bisectors_second_points[: self.params.num_bisectors]

        bisectors_center = (bisectors_first_points + bisectors_second_points) / 2

        # Flip xs with ys and flip sign of on of them to create a 90deg rotation
        inv_bisectors_center_slope = np.fliplr(bisectors_second_points - bisectors_first_points)
        inv_bisectors_center_slope[:, 1] = -inv_bisectors_center_slope[:, 1]

        # Add perpendicular vector to center and normalize
        norm = np.linalg.norm(inv_bisectors_center_slope, axis=1)
        inv_bisectors_center_slope[:, 0] /= norm
        inv_bisectors_center_slope[:, 1] /= norm

        first_bisectors_point = bisectors_center - inv_bisectors_center_slope
        second_bisectors_point = bisectors_center + inv_bisectors_center_slope

        return first_bisectors_point, second_bisectors_point

    def _find_best_intersection(self, fst_points: np.ndarray, sec_points: np.ndarray) -> Tuple[float, float]:
        """fst_points and sec_points are NxD arrays defining N lines. D is the dimension of the space.
            This function returns the least squares intersection of the N lines from the system given by eq. 13 in
            http://cal.cs.illinois.edu/~johannes/research/LS_line_intersecpdf.

        Args:
            fst_points (np.ndarray): First bisectors points.
            sec_points (np.ndarray): Second bisectors points.

        Returns:
            Tuple[float, float]: Best intersection point.

        Reference:
            [1] http://cal.cs.illinois.edu/~johannes/research/LS_line_intersecpdf
        """
        norm_bisectors = (sec_points - fst_points) / np.linalg.norm(sec_points - fst_points, axis=1)[:, np.newaxis]

        # Generate the array of all projectors I - n*n.T
        projections = np.eye(norm_bisectors.shape[1]) - norm_bisectors[:, :, np.newaxis] * norm_bisectors[:, np.newaxis]

        # Generate R matrix and q vector
        R = projections.sum(axis=0)
        q = (projections @ fst_points[:, :, np.newaxis]).sum(axis=0)

        # Solve the least squares problem for the intersection point p: Rp = q
        p = np.linalg.lstsq(R, q, rcond=None)[0]
        intersection_x, intersection_y = p

        return intersection_x.item(), intersection_y.item()
