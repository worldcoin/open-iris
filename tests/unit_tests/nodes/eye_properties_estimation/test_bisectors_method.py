import math
from typing import Tuple

import numpy as np
import pytest

from iris.io.dataclasses import GeometryPolygons
from iris.nodes.eye_properties_estimation.bisectors_method import BisectorsMethod, EyeCentersEstimationError
from tests.unit_tests.utils import generate_arc


@pytest.fixture
def algorithm() -> BisectorsMethod:
    return BisectorsMethod(num_bisectors=100, min_distance_between_sector_points=0.75, max_iterations=50)


def test_calculate_perpendicular_bisectors() -> None:
    radius = 5.0
    center_x, center_y = 0.0, 0.0
    min_distance_between_sector_points_in_px = 5.0

    expected_result_first_bisectors_point = np.array([[1.5, -4.8], [4.1, 2.9]])
    expected_result_second_bisectors_point = np.array([[0.9, -2.9], [2.4, 1.7]])

    algorithm = BisectorsMethod(num_bisectors=2)
    mock_polygons = generate_arc(radius, center_x, center_y, from_angle=0.0, to_angle=2 * np.pi, num_points=10)

    first_bisectors_points, second_bisectors_points = algorithm._calculate_perpendicular_bisectors(
        mock_polygons, min_distance_between_sector_points_in_px
    )

    np.testing.assert_almost_equal(first_bisectors_points, expected_result_first_bisectors_point, decimal=1)
    np.testing.assert_almost_equal(second_bisectors_points, expected_result_second_bisectors_point, decimal=1)


def test_calculate_perpendicular_bisectors_raises_an_exception(algorithm: BisectorsMethod) -> None:
    radius = 5.0
    center_x, center_y = 50.0, 150.0
    min_distance_between_sector_points_in_px = 2 * radius

    mock_polygons = generate_arc(radius, center_x, center_y, from_angle=0.0, to_angle=2 * np.pi)

    with pytest.raises(EyeCentersEstimationError):
        algorithm._calculate_perpendicular_bisectors(mock_polygons, min_distance_between_sector_points_in_px)


@pytest.mark.parametrize(
    "first_bisectors_point,second_bisectors_point,expected_result",
    [
        (np.array([[0.0, -1.0], [0.0, 1.0]]), np.array([[-1.0, 0.0], [1.0, 0.0]]), (0.0, 0.0)),
        (np.array([[-1.0, -1.0], [1.0, 1.0]]), np.array([[-1.0, 1.0], [1.0, -1.0]]), (0.0, 0.0)),
        (np.array([[0.0, -1.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]]), (0.0, 0.0)),
        (
            generate_arc(5.0, 2.5, 2.5, 0.0, 2 * np.pi, num_points=100)[:50],
            generate_arc(5.0, 2.5, 2.5, 0.0, 2 * np.pi, num_points=100)[50:],
            (2.5, 2.5),
        ),
    ],
    ids=["simple", "simple rotated 45", "extremity", "float center"],
)
def test_find_best_intersection(
    algorithm: BisectorsMethod,
    first_bisectors_point: np.ndarray,
    second_bisectors_point: np.ndarray,
    expected_result: Tuple[float, float],
) -> None:
    expected_x, expected_y = expected_result

    result_x, result_y = algorithm._find_best_intersection(first_bisectors_point, second_bisectors_point)

    assert math.isclose(result_x, expected_x, rel_tol=0.1)
    assert math.isclose(result_y, expected_y, rel_tol=0.1)


def test_estimation_on_mock_example(algorithm: BisectorsMethod) -> None:
    pupil_radius = 5.0
    iris_radius = 10.0
    eyeball_radius = 100.0
    pupil_center_x, pupil_center_y = 50.0, 150.0
    iris_center_x, iris_center_y = 55.0, 155.0

    mock_polygons = GeometryPolygons(
        pupil_array=generate_arc(pupil_radius, pupil_center_x, pupil_center_y, 0.0, 2 * np.pi),
        iris_array=generate_arc(iris_radius, iris_center_x, iris_center_y, 0.0, 2 * np.pi),
        eyeball_array=generate_arc(eyeball_radius, iris_center_x, iris_center_y, 0.0, 2 * np.pi),
    )

    result = algorithm(mock_polygons)

    assert math.isclose(result.pupil_x, pupil_center_x, rel_tol=0.1)
    assert math.isclose(result.pupil_y, pupil_center_y, rel_tol=0.1)
    assert math.isclose(result.iris_x, iris_center_x, rel_tol=0.1)
    assert math.isclose(result.iris_y, iris_center_y, rel_tol=0.1)
