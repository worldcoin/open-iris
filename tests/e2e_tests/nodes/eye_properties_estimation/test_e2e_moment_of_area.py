import numpy as np
import pytest

from iris.io.dataclasses import EyeOrientation, GeometryPolygons
from iris.nodes.eye_properties_estimation.moment_of_area import MomentOfArea
from tests.unit_tests.utils import rotated_elliptical_contour


@pytest.mark.parametrize(
    "input_contour,expected_eye_orientation",
    [(rotated_elliptical_contour(theta=0.142857), EyeOrientation(angle=0.142857))],
    ids=["regular"],
)
def test_first_order_area(input_contour: np.ndarray, expected_eye_orientation: EyeOrientation) -> None:
    triangle = np.array([[0, 0], [0, 1], [1, 0]])
    input_geometry_polygon = GeometryPolygons(pupil_array=triangle, iris_array=triangle, eyeball_array=input_contour)

    moments_of_area = MomentOfArea(eccentricity_threshold=0)
    computed_eye_orientaiton = moments_of_area(input_geometry_polygon)

    assert np.abs(computed_eye_orientaiton.angle - expected_eye_orientation.angle) < 1 / 360
