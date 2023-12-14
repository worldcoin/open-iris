from numbers import Number
from typing import Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.dataclasses import GeometryPolygons, IRImage
from iris.io.errors import BoundingBoxEstimationError
from iris.nodes.eye_properties_estimation.iris_bbox_calculator import IrisBBoxCalculator
from tests.unit_tests.utils import generate_arc


@pytest.mark.parametrize(
    "iris_polygon,IRImage_shape,buffer,crop,expected_result",
    [
        (generate_arc(5, 15, 15, 0, 2 * np.pi), (30, 30), 0, True, (10, 20, 10, 20)),
        (generate_arc(5, 15, 15, 0, 2 * np.pi), (30, 30), 5, True, (5, 25, 5, 25)),
        (generate_arc(5, 15, 15, 0, 2 * np.pi), (30, 30), 1.2, True, (9, 21, 9, 21)),
        (generate_arc(5, 15, 15, 0, 2 * np.pi), (30, 30), (5, 2), True, (5, 25, 8, 22)),
        (generate_arc(5, 15, 15, 0, 2 * np.pi), (30, 30), (2.0, 2), True, (5, 25, 8, 22)),
        (generate_arc(5, 15, 15, 0, 2 * np.pi), (30, 30), (2.0, 1.2), True, (5, 25, 9, 21)),
        (generate_arc(5, 5, 5, 0, 2 * np.pi), (10, 10), 0, True, (0, 10, 0, 10)),
        (generate_arc(5, 5, 5, 0, 2 * np.pi), (10, 10), 10, True, (0, 10, 0, 10)),
        (generate_arc(5, 5, 5, 0, 2 * np.pi), (10, 10), 10, False, (-10, 20, -10, 20)),
        (generate_arc(20, 5, 5, 0, 2 * np.pi), (10, 10), 0, True, (0, 10, 0, 10)),
        (generate_arc(15, 5, 5, 0, 2 * np.pi), (10, 10), 0, False, (-10, 20, -10, 20)),
    ],
    ids=[
        "buffer 0",
        "buffer int",
        "buffer float",
        "buffer Tuple[int, int]",
        "buffer Tuple[float, int] (buffer=2 vs buffer=2.0)",
        "buffer Tuple[float, float]",
        "iris touching all corners - buffer 0",
        "iris touching all corners - buffer > 0 overflowing, cropped",
        "iris touching all corners - buffer > 0 overflowing, not cropped",
        "iris overflowing everywhere - buffer = 0, cropped",
        "iris overflowing everywhere - buffer = 0, not cropped",
    ],
)
def test_iris_bbox_calculator(
    iris_polygon: np.ndarray,
    IRImage_shape: Tuple[int, int],
    buffer: Tuple[Number, Number],
    crop: bool,
    expected_result: Tuple[float, float, float, float],
) -> None:
    """Test the iris bounding box calculator."""
    default_polygon = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    geometry_polygons = GeometryPolygons(
        pupil_array=default_polygon, iris_array=iris_polygon, eyeball_array=default_polygon
    )
    ir_image = IRImage(img_data=np.zeros(IRImage_shape), eye_side="left")

    iris_bbox_calcualtor = IrisBBoxCalculator(buffer=buffer, crop=crop)
    iris_bbox = iris_bbox_calcualtor(ir_image, geometry_polygons)

    assert iris_bbox.x_min == expected_result[0]
    assert iris_bbox.x_max == expected_result[1]
    assert iris_bbox.y_min == expected_result[2]
    assert iris_bbox.y_max == expected_result[3]


@pytest.mark.parametrize(
    "buffer,crop",
    [
        (-142857, True),
        (-142.857, True),
        ((-142.857, 1.42857), True),
        (10, "Troue"),
    ],
    ids=[
        "buffer negative int",
        "buffer negative float",
        "buffer tuple with negatiev value",
        "crop string",
    ],
)
def test_iris_bbox_calculator_constructor_raises_exception(buffer: Tuple[Number, Number], crop: bool) -> None:
    """Test the iris bounding box calculator."""
    with pytest.raises(ValidationError):
        _ = IrisBBoxCalculator(buffer=buffer, crop=crop)


@pytest.mark.parametrize(
    "iris_polygon,IRImage_shape,buffer,crop",
    [
        (np.array([[0, 0], [0, 10], [0, 10], [0, 0]]), (30, 30), 0, True),
        (np.array([[0, 0], [10, 0], [10, 0], [0, 0]]), (30, 30), 0, True),
        (np.array([[0, 0], [0, 0], [0, 0]]), (30, 30), 0, True),
        (np.array([[0, 0], [0, 0], [0, 0]]), (30, 30), 0, False),
    ],
    ids=[
        "empty bounding box along x axis",
        "empty bounding box along y axis",
        "empty bounding box",
        "empty bounding box - crop=False",
    ],
)
def test_iris_bbox_calculator_raises_exception(
    iris_polygon: np.ndarray,
    IRImage_shape: Tuple[int, int],
    buffer: Tuple[Number, Number],
    crop: bool,
) -> None:
    """Test the iris bounding box calculator."""
    default_polygon = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
    geometry_polygons = GeometryPolygons(
        pupil_array=default_polygon, iris_array=iris_polygon, eyeball_array=default_polygon
    )
    ir_image = IRImage(img_data=np.zeros(IRImage_shape), eye_side="left")
    iris_bbox_calcualtor = IrisBBoxCalculator(buffer=buffer, crop=crop)
    with pytest.raises(BoundingBoxEstimationError):
        _ = iris_bbox_calcualtor(ir_image, geometry_polygons)
