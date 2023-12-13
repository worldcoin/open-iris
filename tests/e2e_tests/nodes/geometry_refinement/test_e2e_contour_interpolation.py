import os
import pickle
from typing import Any

import numpy as np

from iris.nodes.geometry_refinement.contour_interpolation import ContourInterpolation


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "contour_interpolation")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_e2e_contour_interpolation() -> None:
    mock_input = load_mock_pickle(name="not_interpolated_polygons")

    expected_result = load_mock_pickle(name="e2e_expected_result")

    algorithm = ContourInterpolation(max_distance_between_boundary_points=0.01)
    result = algorithm(polygons=mock_input)

    np.testing.assert_equal(expected_result.pupil_array, result.pupil_array)
    np.testing.assert_equal(expected_result.iris_array, result.iris_array)
    np.testing.assert_equal(expected_result.eyeball_array, result.eyeball_array)
