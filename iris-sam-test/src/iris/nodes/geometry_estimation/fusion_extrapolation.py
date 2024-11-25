from typing import List

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.nodes.geometry_estimation.linear_extrapolation import LinearExtrapolation
from iris.nodes.geometry_estimation.lsq_ellipse_fit_with_refinement import LSQEllipseFitWithRefinement
from iris.utils.math import cartesian2polar


class FusionExtrapolation(Algorithm):
    """Algorithm implements fusion extrapolation that consist of two concreate extrapolation algoriths.
        1) circle extrapolation algorithm - linear extrapolation algorithm
        2) ellipse extrapolation algorithm - least square ellipse fit with iris polygon refinement.

    By default the linear extrapolation algorithm is used but if standard deviation of radiuses is greater than given threshold then least square ellipse fit algorithm is applied because eye is ver likely to be more elliptical then circular.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters of fusion extrapolation algorithm."""

        circle_extrapolation: Algorithm
        ellipse_fit: Algorithm
        algorithm_switch_std_threshold: float

    __parameters_type__ = Parameters

    def __init__(
        self,
        circle_extrapolation: Algorithm = LinearExtrapolation(dphi=360 / 512),
        ellipse_fit: Algorithm = LSQEllipseFitWithRefinement(dphi=360 / 512),
        algorithm_switch_std_threshold: float = 3.5,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            circle_extrapolation (Algorithm, optional): More circular shape estimation algorithm. Defaults to LinearExtrapolation(dphi=360 / 512, degrees / width of normalized image).
            ellipse_fit (Algorithm, optional): More elliptical shape estimation algorithm. Defaults to LSQEllipseFitWithRefinement(dphi=360 / 512, degrees / width of normalized image).
            algorithm_switch_std_threshold (float, optional): Algorithm switch threshold. Defaults to 3.5.
            callbacks (List[Callback], optional): _description_. Defaults to [].
        """
        super().__init__(
            ellipse_fit=ellipse_fit,
            circle_extrapolation=circle_extrapolation,
            algorithm_switch_std_threshold=algorithm_switch_std_threshold,
            callbacks=callbacks,
        )

    def run(self, input_polygons: GeometryPolygons, eye_center: EyeCenters) -> GeometryPolygons:
        """Perform extrapolation algorithm.

        Args:
            input_polygons (GeometryPolygons): Smoothed polygons.
            eye_center (EyeCenters): Computed eye centers.

        Returns:
            GeometryPolygons: Extrapolated polygons
        """
        xs, ys = input_polygons.iris_array[:, 0], input_polygons.iris_array[:, 1]
        rhos, _ = cartesian2polar(xs, ys, eye_center.iris_x, eye_center.iris_y)

        new_poly = self.params.circle_extrapolation(input_polygons, eye_center)

        radius_std = rhos.std()
        if radius_std > self.params.algorithm_switch_std_threshold:
            ellipse_poly = self.params.ellipse_fit(input_polygons)
            new_poly = GeometryPolygons(
                pupil_array=new_poly.pupil_array,
                iris_array=ellipse_poly.iris_array,
                eyeball_array=input_polygons.eyeball_array,
            )

        return new_poly
