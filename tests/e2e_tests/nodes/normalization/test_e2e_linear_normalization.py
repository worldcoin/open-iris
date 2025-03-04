import os
import pickle
from typing import Any

import numpy as np
import pytest
from iris.nodes.normalization.linear_normalization import LinearNormalization


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def algorithm() -> LinearNormalization:
    return LinearNormalization(
        res_in_r=100,
    )


def test_e2e_linear_normalization(algorithm: LinearNormalization) -> None:
    ir_image = load_mock_pickle("ir_image")
    noise_mask = load_mock_pickle("noise_mask")
    eye_orientation = load_mock_pickle("eye_orientation")
    extrapolated_polygons = load_mock_pickle("extrapolated_polygons")

    e2e_expected_result = load_mock_pickle("e2e_expected_result_linear")

    result = algorithm(ir_image, noise_mask, extrapolated_polygons, eye_orientation)

    np.testing.assert_equal(result.normalized_image, e2e_expected_result.normalized_image)
    np.testing.assert_equal(result.normalized_mask, e2e_expected_result.normalized_mask)
