import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import GeometryRefinementError
from iris.nodes.geometry_refinement.contour_points_filter import ContourPointNoiseEyeballDistanceFilter


@pytest.fixture
def algorithm() -> ContourPointNoiseEyeballDistanceFilter:
    return ContourPointNoiseEyeballDistanceFilter(min_distance_to_noise_and_eyeball=0.025)


def test_constructor() -> None:
    mock_min_distance_to_noise_and_eyeball = 0.01

    _ = ContourPointNoiseEyeballDistanceFilter(mock_min_distance_to_noise_and_eyeball)


@pytest.mark.parametrize(
    "min_distance_to_noise_and_eyeball",
    [(-1.0), (0.0)],
    ids=[
        "wrong min_distance_to_noise_and_eyeball < 0",
        "wrong min_distance_to_noise_and_eyeball = 0",
    ],
)
def test_constructor_raises_an_exception(min_distance_to_noise_and_eyeball: float) -> None:
    with pytest.raises(ValidationError):
        _ = ContourPointNoiseEyeballDistanceFilter(min_distance_to_noise_and_eyeball)


def test_filter_polygon_points(algorithm: ContourPointNoiseEyeballDistanceFilter) -> None:
    mock_forbidden_touch_map = np.ones((3, 3)) * np.array([1.0, 0.0, 0.0])
    mock_polygon_points = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ]
    )

    expected_result = np.array(
        [
            [1, 0],
            [1, 1],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
        ]
    )

    result = algorithm._filter_polygon_points(mock_forbidden_touch_map, mock_polygon_points)

    np.testing.assert_equal(result, expected_result)


def test_filter_polygon_points_raises_an_exception(algorithm: ContourPointNoiseEyeballDistanceFilter) -> None:
    mock_forbidden_touch_map = np.ones((3, 3)) * np.array([1.0, 0.0, 0.0])
    mock_polygon_points = np.array(
        [
            [0, 0],
            [0, 1],
            [0, 2],
        ]
    )

    with pytest.raises(GeometryRefinementError):
        _ = algorithm._filter_polygon_points(mock_forbidden_touch_map, mock_polygon_points)
