import math
import os
import pickle
from typing import Any

from iris.nodes.eye_properties_estimation.bisectors_method import BisectorsMethod
from iris.nodes.eye_properties_estimation.pupil_iris_property_calculator import PupilIrisPropertyCalculator


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "bisectors_method")
    mock_path = os.path.join(testdir, f"{name}.pickle")
    return pickle.load(open(mock_path, "rb"))


def test_precomputed_pupil_iris_property() -> None:
    mock_polygons = load_mock_pickle(name="geometry_polygons")

    eye_center_obj = BisectorsMethod()
    eye_center = eye_center_obj(mock_polygons)

    pupil_iris_property_obj = PupilIrisPropertyCalculator()
    p2i_property = pupil_iris_property_obj(mock_polygons, eye_center)

    assert math.isclose(p2i_property.pupil_to_iris_diameter_ratio, 0.543019583685283)
    assert math.isclose(p2i_property.pupil_to_iris_center_dist_ratio, 0.032786957796171405)
