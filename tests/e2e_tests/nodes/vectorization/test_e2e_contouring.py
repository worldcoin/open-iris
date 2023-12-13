import os
import pickle
from typing import Any

import numpy as np
import pytest

from iris.nodes.vectorization.contouring import ContouringAlgorithm, filter_polygon_areas


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "contouring")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def algorithm() -> ContouringAlgorithm:
    return ContouringAlgorithm(contour_filters=[filter_polygon_areas])


def test_e2e_vectorization_algorithm(algorithm: ContouringAlgorithm) -> None:
    mock_geometry_mask = load_mock_pickle(name="geometry_mask")

    expected_result = load_mock_pickle(name="e2e_expected_result")

    result = algorithm(geometry_mask=mock_geometry_mask)

    np.testing.assert_equal(expected_result.pupil_array, result.pupil_array)
    np.testing.assert_equal(expected_result.iris_array, result.iris_array)
    np.testing.assert_equal(expected_result.eyeball_array, result.eyeball_array)
