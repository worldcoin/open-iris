import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from iris.utils import common

MOCK_MASK_SHAPE = (40, 40)


@pytest.fixture
def expected_result() -> np.ndarray:
    expected_result = np.zeros(shape=MOCK_MASK_SHAPE, dtype=bool)
    expected_result[10:21, 10:21] = True
    return expected_result


@pytest.fixture
def expected_result_line() -> np.ndarray:
    expected_result = np.zeros(shape=MOCK_MASK_SHAPE, dtype=bool)
    expected_result[10, 10:21] = True
    return expected_result


@pytest.mark.parametrize(
    "mock_vertices,expected",
    [
        (np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0]]), "expected_result"),
        (np.array([[10.0, 10.0], [20.0, 10.0], [20.0, 20.0], [10.0, 20.0], [10.0, 10.0]]), "expected_result"),
        (np.array([[10.0, 10.0], [20.0, 10.0]]), "expected_result_line"),
        (np.array([[10.0, 10.0], [15.0, 10.0], [20.0, 10.0]]), "expected_result_line"),
    ],
    ids=["standard", "loop", "2 vertices line", "3 vertices line"],
)
def test_contour_to_mask(mock_vertices: np.ndarray, expected: str, request: FixtureRequest) -> None:
    expected_result = request.getfixturevalue(expected)
    result = common.contour_to_mask(mock_vertices, MOCK_MASK_SHAPE)

    assert np.all(expected_result == result)
