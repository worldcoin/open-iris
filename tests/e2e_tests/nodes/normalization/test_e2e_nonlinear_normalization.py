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
    "res_in_r, method",
    [
        (100, NonlinearType.default),
        (100, NonlinearType.wyatt),
    ],
    ids=["default", "wyatt"],
)
def test_getgrids(res_in_r, method) -> None:
    algorithm = NonlinearNormalization(res_in_r=res_in_r, method=method)
    grids30, grids49, grids70 = load_mock_pickle("nonlinear_grids_default")
    results30 = algorithm._getgrids(100, 30, NonlinearType.default)
    results49 = algorithm._getgrids(100, 49, NonlinearType.default)
    results70 = algorithm._getgrids(120, 70, NonlinearType.default)

    np.testing.assert_equal(results30, grids30)
    np.testing.assert_equal(results49, grids49)
    np.testing.assert_equal(results70, grids70)

    grids30_wyatt, grids49_wyatt, grids70_wyatt = load_mock_pickle("nonlinear_grids_wyatt")
    results30 = algorithm._getgrids(100, 30, NonlinearType.wyatt)
    results49 = algorithm._getgrids(100, 49, NonlinearType.wyatt)
    results70 = algorithm._getgrids(120, 70, NonlinearType.wyatt)

    np.testing.assert_equal(results30, grids30_wyatt)
    np.testing.assert_equal(results49, grids49_wyatt)
    np.testing.assert_equal(results70, grids70_wyatt)


@pytest.mark.parametrize(
    "res_in_r, method, expected_filename",
    [
        (100, NonlinearType.default, "e2e_expected_result_nonlinear_default"),
        (100, NonlinearType.wyatt, "e2e_expected_result_nonlinear_wyatt"),
    ],
    ids=["default", "wyatt"],
)
def test_e2e_normalization_nonlinear(res_in_r, method, expected_filename) -> None:
    algorithm = NonlinearNormalization(res_in_r=res_in_r, method=method)
    ir_image = load_mock_pickle("ir_image")
    noise_mask = load_mock_pickle("noise_mask")
    eye_orientation = load_mock_pickle("eye_orientation")
    extrapolated_polygons = load_mock_pickle("extrapolated_polygons")

    e2e_expected_result = load_mock_pickle(expected_filename)

    result = algorithm(ir_image, noise_mask, extrapolated_polygons, eye_orientation)

    np.testing.assert_equal(result.normalized_image, e2e_expected_result.normalized_image)
    np.testing.assert_equal(result.normalized_mask, e2e_expected_result.normalized_mask)
