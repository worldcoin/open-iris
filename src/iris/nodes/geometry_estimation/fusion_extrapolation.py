from typing import List

import numpy as np

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.nodes.geometry_estimation.linear_extrapolation import LinearExtrapolation
from iris.nodes.geometry_estimation.lsq_ellipse_fit_with_refinement import LSQEllipseFitWithRefinement
from iris.utils.math import cartesian2polar


class FusionExtrapolation(Algorithm):
    """Fuse two extrapolation strategies and pick the result based on shape statistics.

    1) Circle-like extrapolation: LinearExtrapolation (linear extrapolation algorithm)
    2) Ellipse-like extrapolation: LSQEllipseFitWithRefinement (least square ellipse fit with iris polygon refinement)

    By default, the linear extrapolation is used. If the (relative) spread of radii
    exceeds a threshold and the ellipse-based result looks sufficiently "regular"
    (based on squared radii ratios), we prefer the ellipse fit.

    Notes:
    - "Relative std" means std/mean with an epsilon guard.
    - We compare regularity via the relative std of iris/pupil squared-radius ratios under three centering choices (circle/circle, ellipse/ellipse, ellipse/circle).
    """

    class Parameters(Algorithm.Parameters):
        """Parameters of fusion extrapolation algorithm."""

        circle_extrapolation: Algorithm
        ellipse_fit: Algorithm
        algorithm_switch_std_threshold: float
        algorithm_switch_std_conditioned_multiplier: float

    __parameters_type__ = Parameters

    def __init__(
        self,
        circle_extrapolation: Algorithm = LinearExtrapolation(dphi=360 / 512),
        ellipse_fit: Algorithm = LSQEllipseFitWithRefinement(dphi=360 / 512),
        algorithm_switch_std_threshold: float = 0.014,
        algorithm_switch_std_conditioned_multiplier: float = 2.0,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            circle_extrapolation (Algorithm, optional): More circular shape estimation algorithm. Defaults to LinearExtrapolation(dphi=360 / 512, degrees / width of normalized image).
            ellipse_fit (Algorithm, optional): More elliptical shape estimation algorithm. Defaults to LSQEllipseFitWithRefinement(dphi=360 / 512, degrees / width of normalized image).
            algorithm_switch_std_threshold (float, optional): Threshold on (rel. std of iris + rel. std of pupil) beyond which we consider switching to ellipse.. Defaults to 0.014.
            algorithm_switch_std_conditioned_multiplier (float, optional): How strongly iris spread penalizes circle regularity in the switch condition. Default: 2.0.
            callbacks (List[Callback], optional): _description_. Defaults to [].
        """
        super().__init__(
            ellipse_fit=ellipse_fit,
            circle_extrapolation=circle_extrapolation,
            algorithm_switch_std_threshold=algorithm_switch_std_threshold,
            algorithm_switch_std_conditioned_multiplier=algorithm_switch_std_conditioned_multiplier,
            callbacks=callbacks,
        )

    @staticmethod
    def _relative_std(x: np.ndarray, eps: float = 1e-12) -> float:
        """
        Relative std = std / max(mean, eps).

        Args:
            x: 1D array-like.
            eps: Small guard to avoid division by zero.

        Returns:
            float: relative std.
        """
        x = np.asarray(x)
        m = float(np.mean(x))
        s = float(np.std(x))
        return s / max(abs(m), eps)

    @staticmethod
    def _squared_relative_radii(
        iris_centered: np.ndarray, pupil_centered: np.ndarray, eps: float = 1e-12
    ) -> np.ndarray:
        """
        Ratio of iris over pupil squared radii, element wise.

        Args:
            iris_centered: (N, 2) centered iris points.
            pupil_centered: (N, 2) centered pupil points.
            eps: Small guard to avoid division by zero.

        Returns:
            (N,) array: iris_sq / pupil_sq
        """
        iris_sq = np.sum(iris_centered**2, axis=1)
        pupil_sq = np.sum(pupil_centered**2, axis=1)
        return np.divide(iris_sq, np.maximum(pupil_sq, eps))

    def run(self, input_polygons: GeometryPolygons, eye_center: EyeCenters) -> GeometryPolygons:
        """Perform extrapolation algorithm and select the most plausible result.

        Args:
            input_polygons (GeometryPolygons): Smoothed polygons.
            eye_center (EyeCenters): Computed eye centers.

        Returns:
            GeometryPolygons: Extrapolated polygons
        """
        xs_iris, ys_iris = input_polygons.iris_array[:, 0], input_polygons.iris_array[:, 1]
        rhos_iris, _ = cartesian2polar(xs_iris, ys_iris, eye_center.iris_x, eye_center.iris_y)
        xs_pupil, ys_pupil = input_polygons.pupil_array[:, 0], input_polygons.pupil_array[:, 1]
        rhos_pupil, _ = cartesian2polar(xs_pupil, ys_pupil, eye_center.pupil_x, eye_center.pupil_y)
        circle_poly = self.params.circle_extrapolation(input_polygons, eye_center)
        ellipse_poly = self.params.ellipse_fit(input_polygons)

        if ellipse_poly is None:
            return circle_poly

        circle_iris = circle_poly.iris_array
        circle_pupil = circle_poly.pupil_array
        ellipse_iris = ellipse_poly.iris_array
        ellipse_pupil = ellipse_poly.pupil_array

        circle_center = np.mean(circle_pupil, axis=0)
        ellipse_center = np.mean(ellipse_pupil, axis=0)

        circle_iris_centered = circle_iris - circle_center
        circle_pupil_centered = circle_pupil - circle_center
        ellipse_iris_centered = ellipse_iris - ellipse_center
        ellipse_pupil_centered = ellipse_pupil - ellipse_center
        ec_iris_centered = ellipse_iris - circle_center

        cc = self._squared_relative_radii(circle_iris_centered, circle_pupil_centered)
        ee = self._squared_relative_radii(ellipse_iris_centered, ellipse_pupil_centered)
        ec = self._squared_relative_radii(ec_iris_centered, circle_pupil_centered)
        cc_reg = self._relative_std(cc)
        ee_reg = self._relative_std(ee)
        ec_reg = self._relative_std(ec)

        radius_std_iris = self._relative_std(rhos_iris)
        radius_std_pupil = self._relative_std(rhos_pupil)
        min_reg = min(ee_reg, ec_reg)

        spread_ok = (radius_std_iris + radius_std_pupil) >= self.params.algorithm_switch_std_threshold
        reg_ok = min_reg <= (cc_reg + radius_std_iris * self.params.algorithm_switch_std_conditioned_multiplier)

        if spread_ok and reg_ok:
            if ee_reg <= ec_reg:
                return GeometryPolygons(
                    pupil_array=ellipse_poly.pupil_array,
                    iris_array=ellipse_poly.iris_array,
                    eyeball_array=input_polygons.eyeball_array,
                )
            else:
                return GeometryPolygons(
                    pupil_array=circle_poly.pupil_array,
                    iris_array=ellipse_poly.iris_array,
                    eyeball_array=input_polygons.eyeball_array,
                )

        return circle_poly
