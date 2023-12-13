import numpy as np
import pytest
from pydantic import ValidationError

from iris.nodes.geometry_refinement.contour_interpolation import ContourInterpolation


@pytest.fixture
def algorithm() -> ContourInterpolation:
    return ContourInterpolation(max_distance_between_boundary_points=0.01)


def test_constructor() -> None:
    mock_max_distance_between_boundary_points = 0.01

    _ = ContourInterpolation(mock_max_distance_between_boundary_points)


@pytest.mark.parametrize(
    "max_distance_between_boundary_points",
    [(-1.0), (0.0)],
    ids=[
        "wrong max_distance_between_boundary_points < 0",
        "wrong max_distance_between_boundary_points = 0",
    ],
)
def test_constructor_raises_an_exception(max_distance_between_boundary_points: float) -> None:
    with pytest.raises(ValidationError):
        _ = ContourInterpolation(max_distance_between_boundary_points)


@pytest.mark.parametrize(
    "mock_polygon,mock_distance_between_points,expected_result",
    [
        (
            np.array([[0.0, 0.0], [50.0, 0.0], [100.0, 0.0]], dtype=np.int32),
            25.0,
            np.array([[0.0, 0.0], [25.0, 0.0], [50.0, 0.0], [75.0, 0.0], [100.0, 0.0]]),
        ),
        (
            np.array([[0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]], dtype=np.int32),
            50.0,
            np.array(
                [[0.0, 0.0], [0.0, 50.0], [0.0, 100.0], [50.0, 100.0], [100.0, 100.0], [100.0, 50.0], [100.0, 0.0]]
            ),
        ),
        (
            np.array([[0.0, 0.0], [0.0, 10.0], [0.0, 15.0]], dtype=np.int32),
            7.0,
            np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 10.0], [0.0, 15.0]]),
        ),
    ],
    ids=["along line", "complex polygon", "not uniform distance"],
)
def test_interpolate_contour_points(
    algorithm: ContourInterpolation,
    mock_polygon: np.ndarray,
    mock_distance_between_points: float,
    expected_result: np.ndarray,
) -> None:
    result = algorithm._interpolate_polygon_points(
        polygon=mock_polygon, max_distance_between_points_px=mock_distance_between_points
    )

    for point in result:
        assert point in expected_result

    for point in expected_result:
        assert point in result
