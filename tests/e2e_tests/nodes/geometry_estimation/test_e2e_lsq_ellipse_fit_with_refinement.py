import os
import pickle
from typing import Any

import numpy as np

from iris.nodes.geometry_estimation.lsq_ellipse_fit_with_refinement import LSQEllipseFitWithRefinement


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_e2e_lsq_ellipse_fit_with_refinement() -> None:
    mock_input = load_mock_pickle(name="smoothing_result")

    expected_result = load_mock_pickle(name="e2e_expected_result_lsq_ellipse_fit_with_refinement")

    algorithm = LSQEllipseFitWithRefinement(dphi=1.0)
    result = algorithm(input_polygons=mock_input)

    np.testing.assert_equal(expected_result.pupil_array, result.pupil_array)
    np.testing.assert_equal(expected_result.iris_array, result.iris_array)
    np.testing.assert_equal(expected_result.eyeball_array, result.eyeball_array)
