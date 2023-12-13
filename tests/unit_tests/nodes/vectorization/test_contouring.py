from functools import partial
from typing import Callable, List

import numpy as np
import pytest
from pydantic import NonNegativeFloat, ValidationError

from iris.io.dataclasses import GeometryMask
from iris.nodes.vectorization.contouring import ContouringAlgorithm, filter_polygon_areas


@pytest.mark.parametrize(
    "mock_polygons,rel_tr,abs_tr,expected_result",
    [
        (
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
                np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),  # area = 0.25
            ],
            0.5,
            0.0,
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
            ],
        ),
        (
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
                np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),  # area = 0.25
            ],
            0.0,
            0.5,
            [
                np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
            ],
        ),
    ],
    ids=["smaller than abs_tr", "smaller than rel_tr"],
)
def test_filter_polygon_areas(
    mock_polygons: List[np.ndarray],
    rel_tr: NonNegativeFloat,
    abs_tr: NonNegativeFloat,
    expected_result: List[np.ndarray],
) -> None:
    result = filter_polygon_areas(mock_polygons, rel_tr=rel_tr, abs_tr=abs_tr)

    np.testing.assert_equal(result, expected_result)


@pytest.fixture
def algorithm() -> ContouringAlgorithm:
    return ContouringAlgorithm(contour_filters=[filter_polygon_areas])


def test_geometry_raster_constructor() -> None:
    mock_pupil_mask = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 1, 1],
        ],
        dtype=bool,
    )

    mock_iris_mask = np.array(
        [
            [0, 0, 0],
            [1, 1, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    mock_eyeball_mask = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    )

    result = GeometryMask(pupil_mask=mock_pupil_mask, iris_mask=mock_iris_mask, eyeball_mask=mock_eyeball_mask)

    np.testing.assert_equal(result.filled_eyeball_mask, mock_eyeball_mask + mock_iris_mask + mock_pupil_mask)
    np.testing.assert_equal(result.filled_iris_mask, mock_iris_mask + mock_pupil_mask)


@pytest.mark.parametrize(
    "mock_pupil_mask,mock_iris_mask,mock_eyeball_mask",
    [
        (
            np.array([0, 0, 0], dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
        ),
        (
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.array([0, 0, 0], dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
        ),
        (
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.ones(shape=(3, 3), dtype=np.uint8),
            np.array([0, 0, 0], dtype=np.uint8),
        ),
    ],
    ids=["wrong dimension of pupil mask", "wrong dimension of iris mask", "wrong dimension of eyeball mask"],
)
def test_geometry_raster_constructor_raises_an_exception(
    mock_pupil_mask: np.ndarray, mock_iris_mask: np.ndarray, mock_eyeball_mask: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        _ = GeometryMask(pupil_mask=mock_pupil_mask, iris_mask=mock_iris_mask, eyeball_mask=mock_eyeball_mask)


@pytest.mark.parametrize(
    "contour_filters",
    [
        ([]),
        ([filter_polygon_areas]),
        ([filter_polygon_areas, partial(filter_polygon_areas, atol=0.1, rtol=0.05)]),
    ],
    ids=["empty filter list", "single element list", "more elements list"],
)
def test_constructor(contour_filters: List[Callable[[List[np.ndarray]], List[np.ndarray]]]) -> None:
    _ = ContouringAlgorithm(contour_filters=contour_filters)


@pytest.mark.parametrize(
    "contour_filters",
    [(None), (filter_polygon_areas)],
    ids=["None value", "func not in a list"],
)
def test_constructor_raises_an_exception(contour_filters: List[Callable[[List[np.ndarray]], List[np.ndarray]]]) -> None:
    with pytest.raises(ValidationError):
        _ = ContouringAlgorithm(contour_filters=contour_filters)


def test_eliminate_tiny_contours() -> None:
    algorithm = ContouringAlgorithm(contour_filters=[partial(filter_polygon_areas, abs_tr=0.5)])

    mock_contours = [
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),  # area = 1.0
        np.array([[0, 0], [0.5, 0], [0.5, 0.5], [0, 0.5]]),  # area = 0.25
    ]

    expected_result = [np.array([[0, 0], [1, 0], [1, 1], [0, 1]])]  # area = 1.0

    # condition: area > 0.5
    result = algorithm._filter_contours(mock_contours)

    np.testing.assert_equal(result, expected_result)
