import numpy as np
import pytest

from iris.nodes.geometry_estimation.linear_extrapolation import LinearExtrapolation
from tests.unit_tests.utils import generate_arc


@pytest.fixture
def algorithm() -> LinearExtrapolation:
    return LinearExtrapolation(dphi=1.0)


@pytest.mark.parametrize(
    "dphi,from_angle,to_angle",
    [
        (1.0, 0.0, 2 * np.pi),
        (1.0, np.pi, 2 * np.pi),
        (1.0, np.pi / 4, 2 * np.pi),
        (1.0, np.pi / 4, 2 * np.pi),
        (1.0, np.pi / 4, 3 * np.pi / 4),
        (1.0, 3 * np.pi / 4, 5 * np.pi / 4),
        (0.1, 0.0, 2 * np.pi),
        (0.1, np.pi, 2 * np.pi),
        (0.1, np.pi / 4, 2 * np.pi),
        (0.1, np.pi / 4, 2 * np.pi),
        (0.1, np.pi / 4, 3 * np.pi / 4),
        (0.1, 3 * np.pi / 4, 5 * np.pi / 4),
    ],
)
def test_estimate_method(dphi: float, from_angle: float, to_angle: float) -> None:
    radius = 1.0
    center_x, center_y = 0.0, 0.0
    num_points = 1000
    algorithm = LinearExtrapolation(dphi)

    mock_arc = generate_arc(radius, center_x, center_y, from_angle, to_angle, num_points)

    # Full circle is expected with same radius, center_x and center_y
    expected_result = generate_arc(radius, center_x, center_y, 0.0, 2 * np.pi, int(360.0 / algorithm.params.dphi))

    result = algorithm._estimate(mock_arc, (center_x, center_y))

    np.testing.assert_allclose(np.sort(result), np.sort(expected_result), atol=1e-1)
