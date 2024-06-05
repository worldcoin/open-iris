import os
import pickle
from typing import Any

import pytest
import numpy as np

from iris.io.dataclasses import EyeOcclusion
from iris.nodes.eye_properties_estimation.occlusion_calculator import OcclusionCalculator


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "occlusion_calculator")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def algorithm() -> OcclusionCalculator:
    return OcclusionCalculator(quantile_angle=30.0)


def test_e2e_occlusion_calculator() -> None:
    mock_extrapolated_polygons = load_mock_pickle(name="extrapolated_polygons")
    mock_noise_mask = load_mock_pickle(name="noise_mask")
    mock_eye_orientation = load_mock_pickle(name="eye_orientation")
    mock_eye_center = load_mock_pickle(name="eye_center")

    algorithm = OcclusionCalculator(quantile_angle=30.0)

    expected_result = EyeOcclusion(visible_fraction=0.9953)

    result = algorithm(mock_extrapolated_polygons, mock_noise_mask, mock_eye_orientation, mock_eye_center)

    np.testing.assert_almost_equal(result.visible_fraction, expected_result.visible_fraction, decimal=4)
