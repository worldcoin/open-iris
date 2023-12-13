import os
import pickle
from typing import Any

import numpy as np

from iris.nodes.geometry_estimation.linear_extrapolation import LinearExtrapolation


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_e2e_linear_extrapolation() -> None:
    mock_input = load_mock_pickle(name="smoothing_result")
    mock_eye_center = load_mock_pickle(name="eye_center")

    expected_result = load_mock_pickle(name="e2e_expected_result_linear_extrapolation")

    algorithm = LinearExtrapolation(dphi=1.0)
    result = algorithm(input_polygons=mock_input, eye_center=mock_eye_center)

    np.testing.assert_equal(expected_result.pupil_array, result.pupil_array)
    np.testing.assert_equal(expected_result.iris_array, result.iris_array)
    np.testing.assert_equal(expected_result.eyeball_array, result.eyeball_array)
