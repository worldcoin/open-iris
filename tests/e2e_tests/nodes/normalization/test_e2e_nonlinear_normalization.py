import os
import pickle
from typing import Any

import numpy as np
import pytest

from iris.nodes.normalization.common import getgrids
from iris.nodes.normalization.nonlinear_normalization import NonlinearNormalization


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "nonlinear_normalization")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def algorithm() -> NonlinearNormalization:
    return NonlinearNormalization(
        res_in_r=100,
    )


def test_getgrids() -> None:
    grids30, grids49, grids70 = load_mock_pickle("nonlinear_grids")
    results30 = getgrids(100, 30)
    results49 = getgrids(100, 49)
    results70 = getgrids(120, 70)

    np.testing.assert_equal(results30, grids30)
    np.testing.assert_equal(results49, grids49)
    np.testing.assert_equal(results70, grids70)


def test_e2e_perspective_normalization_nonlinear(algorithm: NonlinearNormalization) -> None:
    ir_image = load_mock_pickle("ir_image")
    noise_mask = load_mock_pickle("noise_mask")
    eye_orientation = load_mock_pickle("eye_orientation")
    extrapolated_polygons = load_mock_pickle("extrapolated_polygons")

    e2e_expected_result = load_mock_pickle("e2e_expected_result")

    result = algorithm(ir_image, noise_mask, extrapolated_polygons, eye_orientation)

    np.testing.assert_equal(result.normalized_image, e2e_expected_result.normalized_image)
    np.testing.assert_equal(result.normalized_mask, e2e_expected_result.normalized_mask)
