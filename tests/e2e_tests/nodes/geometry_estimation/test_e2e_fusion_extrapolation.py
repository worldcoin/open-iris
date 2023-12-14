import os
import pickle
from typing import Any

import numpy as np

from iris.nodes.geometry_estimation.fusion_extrapolation import FusionExtrapolation


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_e2e_fusion_extrapolation() -> None:
    mock_input = load_mock_pickle(name="smoothing_result")
    mock_eye_center = load_mock_pickle(name="eye_center")

    expected_result = load_mock_pickle(name="e2e_expected_result_fusion_extrapolation")

    algorithm = FusionExtrapolation()
    result = algorithm(input_polygons=mock_input, eye_center=mock_eye_center)

    np.testing.assert_equal(expected_result.pupil_array, result.pupil_array)
    np.testing.assert_equal(expected_result.iris_array, result.iris_array)
    np.testing.assert_equal(expected_result.eyeball_array, result.eyeball_array)
