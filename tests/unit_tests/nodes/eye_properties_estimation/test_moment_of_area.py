import numpy as np
import pytest

from iris.io.dataclasses import GeometryPolygons
from iris.io.errors import EyeOrientationEstimationError
from iris.nodes.eye_properties_estimation.moment_of_area import MomentOfArea
from tests.unit_tests.utils import rotated_elliptical_contour


@pytest.mark.parametrize(
    "input_contour,eccentricity_threshold",
    [(rotated_elliptical_contour(a=1, b=1, theta=0), 0.5)],
    ids=["eccentricity < threshold"],
)
def test_first_order_area_fail_eccentricity_threhsold(input_contour: np.ndarray, eccentricity_threshold: float) -> None:
    triangle = np.array([[0, 0], [0, 1], [1, 0]])
    input_geometry_polygon = GeometryPolygons(pupil_array=triangle, iris_array=triangle, eyeball_array=input_contour)

    with pytest.raises(EyeOrientationEstimationError):
        moments_of_area = MomentOfArea(eccentricity_threshold=eccentricity_threshold)
        moments_of_area(input_geometry_polygon)
