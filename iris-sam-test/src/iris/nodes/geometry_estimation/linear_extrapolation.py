from typing import List, Tuple

import numpy as np
from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.utils import math


class LinearExtrapolation(Algorithm):
    """Implementation of geometry estimation algorithm through linear extrapolation in polar space.

    Algorithm depends on np.interp therefore it's better to perform smoothing beforehand.

    Algorith steps:
        1) Map iris/pupil polygon vertices to polar space based on estimated iris/pupil centers.
        2) For iris/pupil, perform function interpolation in polar space to estimate missing circle points.
            Note: interpolation in polar space is extrapolation in cartesian space.
        3) Take 2 * np.pi / dphi points from function in polar space.
        4) Map iris/pupil points from polar space back to cartesian space.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters of linear extrapolation algorithm."""

        dphi: float = Field(..., gt=0.0, lt=360.0)

    __parameters_type__ = Parameters

    def __init__(self, dphi: float = 0.9, callbacks: List[Callback] = []) -> None:
        """Assign parameters.

        Args:
            dphi (float, optional): phi angle delta used to sample points while doing smoothing by interpolation. Defaults to 0.9.
            callbacks (List[Callback]): callbacks list. Defaults to [].
        """
        super().__init__(dphi=dphi, callbacks=callbacks)

    def run(self, input_polygons: GeometryPolygons, eye_center: EyeCenters) -> GeometryPolygons:
        """Estimate contours.

        Args:
            input_polygons (GeometryPolygons): Input contours.
            eye_center (EyeCenters): Eye's centers.

        Returns:
            GeometryPolygons: Extrapolated contours.
        """
        estimated_pupil = self._estimate(input_polygons.pupil_array, (eye_center.pupil_x, eye_center.pupil_y))
        estimated_iris = self._estimate(input_polygons.iris_array, (eye_center.iris_x, eye_center.iris_y))

        return GeometryPolygons(
            pupil_array=estimated_pupil, iris_array=estimated_iris, eyeball_array=input_polygons.eyeball_array
        )

    def _estimate(self, vertices: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
        """Estimate a circle fit for a single contour.

        Args:
            vertices (np.ndarray): Contour's vertices.
            center_xy (Tuple[float, float]): Contour's center position.

        Returns:
            np.ndarray: Estimated polygon.
        """
        rhos, phis = math.cartesian2polar(vertices[:, 0], vertices[:, 1], *center_xy)

        padded_rhos = np.concatenate([rhos, rhos, rhos])
        padded_phis = np.concatenate([phis - 2 * np.pi, phis, phis + 2 * np.pi])

        interpolated_phis = np.arange(padded_phis.min(), padded_phis.max(), np.radians(self.params.dphi))
        interpolated_rhos = np.interp(interpolated_phis, xp=padded_phis, fp=padded_rhos, period=2 * np.pi)

        mask = (interpolated_phis >= 0) & (interpolated_phis < 2 * np.pi)
        interpolated_phis, interpolated_rhos = interpolated_phis[mask], interpolated_rhos[mask]

        xs, ys = math.polar2cartesian(interpolated_rhos, interpolated_phis, *center_xy)
        estimated_vertices = np.column_stack([xs, ys])

        return estimated_vertices
