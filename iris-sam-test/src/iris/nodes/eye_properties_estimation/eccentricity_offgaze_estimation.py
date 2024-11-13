from typing import List, Literal

import cv2
import numpy as np

import iris.utils.math as math_utils
from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import GeometryPolygons, Offgaze


def get_eccentricity_through_moments(shape_array: np.ndarray) -> float:
    """Determine the eccentricity of the shape through its second order image moments.

    Args:
        shape_array (np.ndarray): Shape array.

    Returns:
        float: Computed eccentricity.
    """
    moments = cv2.moments(shape_array)
    eccentricity = math_utils.eccentricity(moments)
    return eccentricity


def get_eccentricity_through_ellipse_fit(shape_array: np.ndarray) -> float:
    """Determine the eccentricity of the shape by fitting an ellipse (default method).

    Args:
        shape_array (np.ndarray): Shape array.

    Returns:
        float: Computed eccentricity.
    """
    ellipse = cv2.fitEllipse(shape_array)
    eccentricity = np.sqrt(1 - (ellipse[1][0] / ellipse[1][1]) ** 2)
    return eccentricity


def get_eccentricity_through_ellipse_fit_direct(shape_array: np.ndarray) -> float:
    """Determine the eccentricity of the shape by fitting an ellipse (DIRECT method).

    Args:
        shape_array (np.ndarray): Shape array.

    Returns:
        float: Computed eccentricity.
    """
    ellipse = cv2.fitEllipseDirect(shape_array)
    eccentricity = np.sqrt(1 - (ellipse[1][0] / ellipse[1][1]) ** 2)
    return eccentricity


def get_eccentricity_through_ellipse_fit_ams(shape_array: np.ndarray) -> float:
    """Determine the eccentricity of the shape by fitting an ellipse (AMS method).

    Args:
        shape_array (np.ndarray): Shape array.

    Returns:
        float: Computed eccentricity.
    """
    ellipse = cv2.fitEllipseAMS(shape_array)
    eccentricity = np.sqrt(1 - (ellipse[1][0] / ellipse[1][1]) ** 2)
    return eccentricity


class EccentricityOffgazeEstimation(Algorithm):
    """Determines an off-gaze score by assembling the eccentricity of the iris and pupil polygons.

    The goal of this algorithm is to attribute an offgaze score to the GeometryPolygons associated with an IR image.

    The user's iris is approximated to a circle.
    If the plane containing that circle (i.e. the iris) is orthogonal to the direction of the camera
    (i.e. the user looks straight at the camera), then the iris will appear as circular as possible.
    If the user doesn't look straight at the camera, the circle will appear tilted because of the perspective. The iris
    (and pupil) circle will then tend towards an ellipsis more than a circle.

    This shift in general shape can be measured through the eccentricity of the iris (resp. pupil), a measure using
    moments of the shape, between 0 (perfect circle) and 1 (perfect line)

    The eccentricity of a shape can be determined in two ways:
      * `moments`: the moments of the shape are determined, from which we can infer its eccentricity
      * `fit_ellipse`: an ellipse is fitted to the shape, and the ellipse's eccentricity is returned
      * `fit_ellipse_direct`: TBFU
      * `fit_ellipse_ams`: TBFU
    The eccentricity of the iris and pupil can then be assembled in various ways:
      * `min`: the minimum between pupil and iris eccentricity
      * `max`: the maximum between pupil and iris eccentricity
      * `mean`: the average between pupil and iris eccentricity
      * `only_pupil`: pupil eccentricity
      * `only_iris`: iris eccentricity

    LIMITATIONS:

    It is known that irises (and pupil even more) are not perfectly circular shapes. This means that non-offgaze might
    have significantly non-zero values. Also, it is possible that an offgaze might have a pupil and / or iris
    very circular.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for EccentricityOffgazeEstimation objects."""

        assembling_method: Literal["min", "max", "mean", "only_pupil", "only_iris"]
        eccentricity_method: Literal["moments", "ellipse_fit", "ellipse_fit_direct", "ellipse_fit_ams"]

    __parameters_type__ = Parameters

    eccentricity_method2function_mapping = {
        "moments": get_eccentricity_through_moments,
        "ellipse_fit": get_eccentricity_through_ellipse_fit,
        "ellipse_fit_direct": get_eccentricity_through_ellipse_fit_direct,
        "ellipse_fit_ams": get_eccentricity_through_ellipse_fit_ams,
    }

    assembling_method2function_mapping = {
        "min": min,
        "max": max,
        "mean": lambda x, y: (x + y) / 2,
        "only_pupil": lambda x, _: x,
        "only_iris": lambda _, y: y,
    }

    def __init__(
        self,
        assembling_method: Literal["min", "max", "mean", "only_pupil", "only_iris"] = "min",
        eccentricity_method: Literal["moments", "ellipse_fit", "ellipse_fit_direct", "ellipse_fit_ams"] = "moments",
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            assembling_method (Literal["min", "max", "mean", "only_pupil", "only_iris"], optional): How are the pupil eccentricity and iris eccentricity assembled. Defaults to "min".
            eccentricity_method (Literal["moments", "ellipse_fit", "ellipse_fit_direct", "ellipse_fit_ams"], optional): How is the eccentricity determined. Defaults to "moments".
            callbacks (List[Callback]): callbacks list. Defaults to [].
        """
        super().__init__(
            assembling_method=assembling_method, eccentricity_method=eccentricity_method, callbacks=callbacks
        )

    def run(self, geometries: GeometryPolygons) -> Offgaze:
        """Calculate offgaze estimation.

        Args:
            geometries (GeometryPolygons): polygons used for offgaze estimation.

        Returns:
            Offgaze: offgaze object.
        """
        eccentricity_function = EccentricityOffgazeEstimation.eccentricity_method2function_mapping[
            self.params.eccentricity_method
        ]

        assembling_function = EccentricityOffgazeEstimation.assembling_method2function_mapping[
            self.params.assembling_method
        ]

        pupil_eccentricity = eccentricity_function(geometries.pupil_array)
        iris_eccentricity = eccentricity_function(geometries.iris_array)

        offgaze = Offgaze(score=assembling_function(pupil_eccentricity, iris_eccentricity))
        return offgaze
