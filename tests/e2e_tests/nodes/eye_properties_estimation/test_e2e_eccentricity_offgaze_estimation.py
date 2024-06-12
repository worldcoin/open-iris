from typing import Literal

import numpy as np
import pytest

from iris.io.dataclasses import GeometryPolygons
from iris.nodes.eye_properties_estimation.eccentricity_offgaze_estimation import EccentricityOffgazeEstimation
from tests.unit_tests.utils import rotated_elliptical_contour


@pytest.mark.parametrize(
    "geometry_polygons,assembling_method,eccentricity_method,expected_eccentricity",
    [
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0.838,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=-np.pi / 2, centered=True),
                iris_array=rotated_elliptical_contour(a=5, b=1, theta=-np.pi / 2, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0.838,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0.142857, centered=True),
                iris_array=rotated_elliptical_contour(a=5, b=1, theta=0.142857, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0.838,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=2 * rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0.838,
        ),
        (
            GeometryPolygons(
                pupil_array=0.1 * rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=10 * rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0.838,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=1, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=2, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0,
        ),
        (
            GeometryPolygons(
                pupil_array=np.array([[0, 0], [1, 0], [0.5, 2e-2]]),
                iris_array=np.array([[0, 0], [1, 0], [0.5, 2e-2]]),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0.997,
        ),
        (
            GeometryPolygons(
                pupil_array=np.array([[0, 0], [1, 0]]),
                iris_array=np.array([[0, 0], [1, 0]]),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            1.0,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "moments",
            0.341,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "max",
            "moments",
            0.838,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "mean",
            "moments",
            0.589,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "only_iris",
            "moments",
            0.341,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=5, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "only_pupil",
            "moments",
            0.838,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "ellipse_fit",
            0.859,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=1, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=1, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "ellipse_fit",
            0.0,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "ellipse_fit_direct",
            0.859,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=1, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=1, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "ellipse_fit_direct",
            0.0,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=2, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "ellipse_fit_ams",
            0.860,
        ),
        (
            GeometryPolygons(
                pupil_array=rotated_elliptical_contour(a=1, b=1, theta=0, centered=True),
                iris_array=rotated_elliptical_contour(a=1, b=1, theta=0, centered=True),
                eyeball_array=np.zeros((3, 2)),
            ),
            "min",
            "ellipse_fit_ams",
            0.0,
        ),
    ],
    ids=[
        "Same ellipse, iris & pupil have the same angle (angle 1/3) - min",
        "Same ellipse, iris & pupil have the same angle (angle 2/3) - min",
        "Same ellipse, iris & pupil have the same angle (angle 3/3) - min",
        "Same ellipse but iris & pupil have different size (ratio 1/2) - min",
        "Same ellipse but iris & pupil have different size (ratio 2/2) - min",
        "From circle to line (eccentricity = 0) - min",
        "From circle to line (eccentricity = 0.997) - min",
        "From circle to line (eccentricity = 1) - min",
        "Different eccentricities - min",
        "Different eccentricities - max",
        "Different eccentricities - mean",
        "Different eccentricities - only_pupil",
        "Different eccentricities - only_iris",
        "Ellipse fit (eccentricity ~= sqrt(3)/2)",
        "Ellipse fit (eccentricity = 0.)",
        "Ellipse fit Direct (eccentricity ~= sqrt(3)/2)",
        "Ellipse fit Direct (eccentricity = 0.)",
        "Ellipse fit AMS (eccentricity ~= sqrt(3)/2)",
        "Ellipse fit AMS (eccentricity = 0.)",
    ],
)
def test_eccentricity_offgaze_estimation(
    geometry_polygons: GeometryPolygons,
    assembling_method: Literal["min", "max", "mean", "only_pupil", "only_iris"],
    eccentricity_method: Literal["moments", "fit_ellipse"],
    expected_eccentricity: float,
) -> None:
    offgaze_estimator = EccentricityOffgazeEstimation(
        assembling_method=assembling_method, eccentricity_method=eccentricity_method
    )

    offgaze = offgaze_estimator(geometry_polygons)

    assert np.abs(offgaze.score - expected_eccentricity) < 1e-3
