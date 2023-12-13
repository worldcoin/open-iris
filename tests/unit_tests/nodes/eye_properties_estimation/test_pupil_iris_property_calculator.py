import math

import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.dataclasses import GeometryPolygons
from iris.nodes.eye_properties_estimation.bisectors_method import BisectorsMethod
from iris.nodes.eye_properties_estimation.pupil_iris_property_calculator import (
    PupilIrisPropertyCalculator,
    PupilIrisPropertyEstimationError,
)


@pytest.mark.parametrize(
    "min_iris_diameter",
    [
        pytest.param(0),
        pytest.param("a"),
        pytest.param(-10),
        pytest.param(np.ones(3)),
    ],
    ids=[
        "min_iris_diameter should be larger than zero",
        "min_iris_diameter should be float",
        "min_iris_diameter should not be negative",
        "min_iris_diameter should not be array",
    ],
)
def test_iris_diameter_threshold_raises_an_exception(min_iris_diameter: float) -> None:
    with pytest.raises(ValidationError):
        _ = PupilIrisPropertyCalculator(min_iris_diameter=min_iris_diameter)


@pytest.mark.parametrize(
    "min_pupil_diameter",
    [
        pytest.param(0),
        pytest.param("[]"),
        pytest.param(-1),
        pytest.param(np.zeros((2, 2))),
    ],
    ids=[
        "min_pupil_diameter should be larger than zero",
        "min_pupil_diameter should be float",
        "min_pupil_diameter should not be negative",
        "min_pupil_diameter should not be array",
    ],
)
def test_pupil_diameter_threshold_raises_an_exception(min_pupil_diameter: float) -> None:
    with pytest.raises(ValidationError):
        _ = PupilIrisPropertyCalculator(min_pupil_diameter=min_pupil_diameter)


@pytest.mark.parametrize(
    "min_pupil_diameter, min_iris_diameter, pupil_array, iris_array",
    [
        pytest.param(
            100,
            450,
            np.array([[0, 0], [300, 0], [150, np.sqrt(3) * 150]]),
            np.array([[0, 0], [0, 300], [300, 300], [300, 0]]),
        ),
        pytest.param(
            20,
            300,
            np.array([[0, 0], [0, 10], [10, 10], [10, 0]]),
            np.array([[0, 0], [0, 300], [300, 300], [300, 0]]),
        ),
    ],
    ids=[
        "iris diameter is less than min_iris_diameter",
        "pupil diameter is smaller than min_pupil_diameter",
    ],
)
def test_pupil_iris_property_raise_an_exception(
    min_pupil_diameter: float, min_iris_diameter: float, pupil_array: np.ndarray, iris_array: np.ndarray
) -> None:
    mock_polygons = GeometryPolygons(pupil_array=pupil_array, iris_array=iris_array, eyeball_array=iris_array)
    eye_center_obj = BisectorsMethod()
    eye_center = eye_center_obj(mock_polygons)
    pupil_iris_property_obj = PupilIrisPropertyCalculator(
        min_pupil_diameter=min_pupil_diameter, min_iris_diameter=min_iris_diameter
    )
    with pytest.raises(PupilIrisPropertyEstimationError):
        _ = pupil_iris_property_obj(mock_polygons, eye_center)


@pytest.mark.parametrize(
    "min_pupil_diameter, min_iris_diameter, pupil_array, iris_array, expected_diameter_ratio, expected_center_dist_ratio",
    [
        pytest.param(
            20,
            300,
            np.array([[0, 0], [300, 0], [150, np.sqrt(3) * 150]]),
            np.array([[0, 0], [0, 300], [300, 300], [300, 0]]),
            0.7071067813672357,
            0.2988585032058934,
        ),
        pytest.param(
            100,
            300,
            np.array([[0, 0], [0, 200], [200, 200], [200, 0]]),
            np.array([[0, 0], [0, 300], [300, 300], [300, 0]]),
            2 / 3,
            1 / 3,
        ),
        pytest.param(
            0.1,
            0.5,
            np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
            np.array([[0, 0], [0, 200], [200, 200], [200, 0]]),
            1 / 200,
            0.995,
        ),
    ],
    ids=["regular1", "regular2", "regular3"],
)
def test_pupil_iris_property(
    min_pupil_diameter: float,
    min_iris_diameter: float,
    pupil_array: np.ndarray,
    iris_array: np.ndarray,
    expected_diameter_ratio: float,
    expected_center_dist_ratio: float,
) -> None:
    mock_polygons = GeometryPolygons(pupil_array=pupil_array, iris_array=iris_array, eyeball_array=iris_array)
    eye_center_obj = BisectorsMethod()
    eye_center = eye_center_obj(mock_polygons)
    pupil_iris_property_obj = PupilIrisPropertyCalculator(
        min_pupil_diameter=min_pupil_diameter, min_iris_diameter=min_iris_diameter
    )
    p2i_property = pupil_iris_property_obj(mock_polygons, eye_center)
    assert math.isclose(p2i_property.pupil_to_iris_diameter_ratio, expected_diameter_ratio, rel_tol=1e-03)
    assert math.isclose(p2i_property.pupil_to_iris_center_dist_ratio, expected_center_dist_ratio, rel_tol=1e-03)
