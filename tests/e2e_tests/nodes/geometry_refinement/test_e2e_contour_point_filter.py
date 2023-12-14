import os
import pickle
from typing import Any

import numpy as np

from iris.nodes.geometry_refinement.contour_points_filter import ContourPointNoiseEyeballDistanceFilter


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "contour_point_filter")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_e2e_contour_point_filter() -> None:
    mock_input = load_mock_pickle(name="interpolated_polygons")
    mock_noise_mask = load_mock_pickle(name="noise_mask")

    expected_result = load_mock_pickle(name="e2e_expected_result")

    algorithm = ContourPointNoiseEyeballDistanceFilter(min_distance_to_noise_and_eyeball=0.025)
    result = algorithm(polygons=mock_input, geometry_mask=mock_noise_mask)

    np.testing.assert_equal(expected_result.pupil_array, result.pupil_array)
    np.testing.assert_equal(expected_result.iris_array, result.iris_array)
    np.testing.assert_equal(expected_result.eyeball_array, result.eyeball_array)
