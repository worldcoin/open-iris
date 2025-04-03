import os
import pickle
from typing import Any

import numpy as np
import pytest

from iris.nodes.normalization.nonlinear_normalization import NonlinearNormalization, NonlinearType


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.mark.parametrize(
    "res_in_r, method, expected_grids_filename",
    [
        (100, NonlinearType.default, "nonlinear_grids_default"),
        (100, NonlinearType.wyatt, "nonlinear_grids_wyatt"),
    ],
    ids=["default_grids", "wyatt_grids"],
)
def test_getgrids(res_in_r: int, method: NonlinearType, expected_grids_filename: str) -> None:
    algorithm = NonlinearNormalization(res_in_r=res_in_r, method=method)
    grids30, grids49, grids70 = load_mock_pickle(expected_grids_filename)
    results30 = algorithm._getgrids(100, 30, method)
    results49 = algorithm._getgrids(100, 49, method)
    results70 = algorithm._getgrids(120, 70, method)

    np.testing.assert_allclose(results30, grids30, rtol=1e-08)
    np.testing.assert_allclose(results49, grids49, rtol=1e-08)
    np.testing.assert_allclose(results70, grids70, rtol=1e-08)


@pytest.mark.parametrize(
    "res_in_r, method, expected_filename",
    [
        (100, NonlinearType.default, "e2e_expected_result_nonlinear_default"),
        (100, NonlinearType.wyatt, "e2e_expected_result_nonlinear_wyatt"),
    ],
    ids=["default", "wyatt"],
)
def test_e2e_normalization_nonlinear(res_in_r: int, method: NonlinearType, expected_filename: str) -> None:
    algorithm = NonlinearNormalization(res_in_r=res_in_r, method=method)
    ir_image = load_mock_pickle("ir_image")
    noise_mask = load_mock_pickle("noise_mask")
    eye_orientation = load_mock_pickle("eye_orientation")
    extrapolated_polygons = load_mock_pickle("extrapolated_polygons")

    e2e_expected_result = load_mock_pickle(expected_filename)

    result = algorithm(ir_image, noise_mask, extrapolated_polygons, eye_orientation)

    np.testing.assert_equal(result.normalized_image, e2e_expected_result.normalized_image)
    np.testing.assert_equal(result.normalized_mask, e2e_expected_result.normalized_mask)
