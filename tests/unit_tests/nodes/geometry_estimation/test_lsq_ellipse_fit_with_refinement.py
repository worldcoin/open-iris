import numpy as np
import pytest

from iris.nodes.geometry_estimation.lsq_ellipse_fit_with_refinement import LSQEllipseFitWithRefinement


@pytest.fixture
def algorithm() -> LSQEllipseFitWithRefinement:
    return LSQEllipseFitWithRefinement()


@pytest.mark.parametrize(
    "src_pt, dst_pts, expected_index",
    [
        (np.array([0.0, 0.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 0),
        (np.array([1.0, 1.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 1),
        (np.array([2.0, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 2),
        (np.array([0.2, 0.2]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 0),
        (np.array([0.8, 0.8]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 1),
        (np.array([1.4, 1.4]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 1),
        (np.array([1.9, 1.6]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 2),
        (np.array([3.4, 3.4]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]), 2),
    ],
)
def test_find_correspondence(
    algorithm: LSQEllipseFitWithRefinement, src_pt: np.ndarray, dst_pts: np.ndarray, expected_index: int
) -> None:
    result_idx = algorithm._find_correspondence(src_pt, dst_pts)

    assert result_idx == expected_index
