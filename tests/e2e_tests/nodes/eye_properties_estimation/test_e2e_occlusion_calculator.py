import os
import pickle
from typing import Any

import numpy as np
import pytest

from iris.io.dataclasses import EyeOcclusion
from iris.nodes.eye_properties_estimation.occlusion_calculator import OcclusionCalculator


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "occlusion_calculator")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def algorithm() -> OcclusionCalculator:
    return OcclusionCalculator(quantile_angle=30.0)


@pytest.mark.parametrize(
    "extrapolated_polygons_name, noise_mask_name, eye_orientation_name, eye_center_name, expected_result",
    [
        (
            "extrapolated_polygons_1",
            "noise_mask_1",
            "eye_orientation_1",
            "eye_center_1",
            EyeOcclusion(visible_fraction=0.9953),
        ),
        (
            "extrapolated_polygons_2",
            "noise_mask_2",
            "eye_orientation_2",
            "eye_center_2",
            EyeOcclusion(visible_fraction=0.9904),
        ),
        (
            "extrapolated_polygons_cropped",
            "noise_mask_cropped",
            "eye_orientation_cropped",
            "eye_center_cropped",
            EyeOcclusion(visible_fraction=0.5652),
        ),
    ],
    ids=["regular 1", "regular 2", "heavily cropped iris"],
)
def test_e2e_occlusion_calculator(
    extrapolated_polygons_name, noise_mask_name, eye_orientation_name, eye_center_name, expected_result
) -> None:
    mock_extrapolated_polygons = load_mock_pickle(extrapolated_polygons_name)
    mock_noise_mask = load_mock_pickle(noise_mask_name)
    mock_eye_orientation = load_mock_pickle(eye_orientation_name)
    mock_eye_center = load_mock_pickle(eye_center_name)

    algorithm = OcclusionCalculator(quantile_angle=30.0)

    result = algorithm(mock_extrapolated_polygons, mock_noise_mask, mock_eye_orientation, mock_eye_center)

    np.testing.assert_almost_equal(result.visible_fraction, expected_result.visible_fraction, decimal=4)
