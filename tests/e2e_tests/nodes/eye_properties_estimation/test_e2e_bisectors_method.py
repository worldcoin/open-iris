import math
import os
import pickle
from typing import Any

import pytest

from iris.nodes.eye_properties_estimation.bisectors_method import BisectorsMethod


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "bisectors_method")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def algorithm() -> BisectorsMethod:
    return BisectorsMethod(num_bisectors=100, min_distance_between_sector_points=0.75, max_iterations=50)


def test_e2e_bisectors_method_algorithm(algorithm: BisectorsMethod) -> None:
    mock_polygons = load_mock_pickle(name="geometry_polygons")
    expected_result = load_mock_pickle(name="e2e_expected_result")

    result = algorithm(geometries=mock_polygons)

    assert math.isclose(result.pupil_x, expected_result.pupil_x)
    assert math.isclose(result.pupil_y, expected_result.pupil_y)
    assert math.isclose(result.iris_x, expected_result.iris_x)
    assert math.isclose(result.iris_y, expected_result.iris_y)
