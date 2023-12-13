import os
import pickle
from typing import Any

import numpy as np

from iris.nodes.geometry_refinement.smoothing import Smoothing


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "smoothing")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_e2e_smoothing() -> None:
    mock_input = load_mock_pickle(name="before_smoothing")
    mock_eye_center = load_mock_pickle(name="eye_center")

    expected_result = load_mock_pickle(name="e2e_expected_result")

    algorithm = Smoothing(dphi=1, kernel_size=10)
    result = algorithm(polygons=mock_input, eye_centers=mock_eye_center)

    np.testing.assert_equal(expected_result.pupil_array, result.pupil_array)
    np.testing.assert_equal(expected_result.iris_array, result.iris_array)
    np.testing.assert_equal(expected_result.eyeball_array, result.eyeball_array)
