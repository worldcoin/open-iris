from typing import Tuple

import numpy as np
import pytest

from iris.io.errors import GeometryRefinementError
from iris.nodes.geometry_refinement.smoothing import Smoothing
from tests.unit_tests.utils import generate_arc


@pytest.fixture
def algorithm() -> Smoothing:
    return Smoothing(dphi=1, kernel_size=10)


@pytest.mark.parametrize(
    "arc,expected_num_gaps",
    [
        # on arc - no gaps
        (generate_arc(10, 0.0, 0.0, 0.0, 2 * np.pi), 0),
        # one arc - one gap
        (generate_arc(10, 0.0, 0.0, 0.0, np.pi), 1),
        # one arc - one gap not at 0/2pi
        (generate_arc(10, 0.0, 0.0, np.pi, 2.5 * np.pi), 1),
        # two arcs - two gaps
        (np.vstack([generate_arc(10, 0.0, 0.0, 0.0, np.pi / 4), generate_arc(10, 0.0, 0.0, np.pi, 4 / 3 * np.pi)]), 2),
        # one arc in two parts - one gap
        (
            np.vstack(
                [generate_arc(10, 0.0, 0.0, 0.0, np.pi / 4), generate_arc(10, 0.0, 0.0, 4 / 3 * np.pi, 2 * np.pi)]
            ),
            1,
        ),
        # one arc shuffled - one gap
        (np.random.permutation(generate_arc(10, 0.0, 0.0, 0.0, np.pi)), 1),
    ],
)
def test_cut_into_arcs(algorithm: Smoothing, arc: np.ndarray, expected_num_gaps: int) -> None:
    center_x, center_y = 0.0, 0.0

    result_arcs, result_num_gaps = algorithm._cut_into_arcs(arc, (center_x, center_y))

    # Ensure number of gaps is as expected.
    assert result_num_gaps == expected_num_gaps

    # Ensure that both arrays have the exact same elements.
    arc_sorted = arc[np.lexsort(arc.T)]
    result_stacked = np.vstack(result_arcs)
    result_sorted = result_stacked[np.lexsort(result_stacked.T)]
    np.testing.assert_equal(arc_sorted, result_sorted)


@pytest.mark.parametrize(
    "phis, rhos, expected_result",
    [
        (
            np.array([0.0, 0.02621434, 0.05279587, 0.08517275, 0.12059719, 0.15643903]),
            np.array([36.89178243, 36.62426603, 36.38227748, 37.14610941, 36.90603523, 36.71284955]),
            (
                np.array(
                    [
                        0.0,
                        0.01745329,
                        0.03490659,
                        0.05235988,
                        0.06981317,
                        0.08726646,
                        0.10471976,
                        0.12217305,
                        0.13962634,
                    ]
                ),
                np.array(
                    [
                        36.80346909,
                        36.89178243,
                        36.89178243,
                        36.80346909,
                        36.80346909,
                        36.80346909,
                        36.78374777,
                        36.78374777,
                        36.78374777,
                    ]
                ),
            ),
        )
    ],
)
def test_smooth_array(
    algorithm: Smoothing, phis: np.ndarray, rhos: np.ndarray, expected_result: Tuple[np.ndarray, np.ndarray]
) -> None:
    result = algorithm._smooth_array(phis, rhos)
    np.testing.assert_almost_equal(expected_result, result, decimal=0)


def test_smooth_arc(algorithm: Smoothing) -> None:
    center_x, center_y = 0.0, 0.0

    mock_arc = generate_arc(10, center_x, center_y, 0.0, np.pi, 180)
    expected_result = mock_arc[algorithm.kernel_offset : -algorithm.kernel_offset]

    result = algorithm._smooth_arc(mock_arc, (center_x, center_y))

    np.testing.assert_almost_equal(expected_result, result, decimal=0)


def test_smooth_circular_shape(algorithm: Smoothing) -> None:
    center_x, center_y = 0.0, 0.0

    mock_arc = generate_arc(10, center_x, center_y, 0.0, 2 * np.pi, 1000)
    expected_result = generate_arc(10, center_x, center_y, 0.0, 2 * np.pi, 360)

    result = algorithm._smooth_circular_shape(mock_arc, (center_x, center_y))

    np.testing.assert_almost_equal(expected_result, result, decimal=0)


def test_sort_two_arrays(algorithm: Smoothing) -> None:
    first_array = np.array([3.0, 2.0, 1.0])
    second_array = np.array([1.0, 2.0, 3.0])

    first_sorted, second_sorted = algorithm._sort_two_arrays(first_array, second_array)

    np.testing.assert_equal(first_sorted, second_array)
    np.testing.assert_equal(second_sorted, first_array)


def test_rolling_median(algorithm: Smoothing) -> None:
    signal = np.arange(0, 10, 1)
    kernel_offset = 3

    expected_result = np.array([3.0, 4.0, 5.0, 6.0])

    result = algorithm._rolling_median(signal, kernel_offset)

    np.testing.assert_equal(result, expected_result)


def test_rolling_median_raises_an_error_when_not_1D_signal(algorithm: Smoothing) -> None:
    signal = np.arange(0, 10, 1).reshape((5, 2))
    kernel_offset = 3

    with pytest.raises(GeometryRefinementError):
        _ = algorithm._rolling_median(signal, kernel_offset)


def test_find_start_index_raises_an_error_when_phis_not_sorted_ascendingly(algorithm: Smoothing) -> None:
    mock_phis = np.arange(0, 100, 1)
    np.random.shuffle(mock_phis)

    with pytest.raises(GeometryRefinementError):
        _ = algorithm._find_start_index(mock_phis)
