import os
import pickle
from typing import Any

import numpy as np
from iris.nodes.normalization.utils import getgrids


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_getgrids() -> None:
    grids30, grids49, grids70 = load_mock_pickle("nonlinear_grids")
    results30 = getgrids(100, 30)
    results49 = getgrids(100, 49)
    results70 = getgrids(120, 70)

    np.testing.assert_equal(results30, grids30)
    np.testing.assert_equal(results49, grids49)
    np.testing.assert_equal(results70, grids70)
