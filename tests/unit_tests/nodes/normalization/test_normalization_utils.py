from typing import Tuple

import numpy as np
import pytest

from iris.nodes.normalization.utils import correct_orientation, interpolate_pixel_intensity, to_uint8
from tests.unit_tests.utils import generate_arc


@pytest.mark.parametrize(
    "eye_orientation,pupil_points,iris_points,expected_pupil_points,expected_iris_points",
    [
        (
            -1.0,
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360),
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360), 1, axis=0),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 360), 1, axis=0),
        ),
        (
            -1.0,
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720),
            generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720), 2, axis=0),
            np.roll(generate_arc(10.0, 0.0, 0.0, 0.0, 2 * np.pi, 720), 2, axis=0),
        ),
    ],
    ids=["1 point rotation", "2 points rotation"],
)
def test_correct_orientation(
    eye_orientation: float,
    pupil_points: np.ndarray,
    iris_points: np.ndarray,
    expected_pupil_points: np.ndarray,
    expected_iris_points: np.ndarray,
) -> None:
    result_pupil_points, result_iris_points = correct_orientation(
        pupil_points=pupil_points,
        iris_points=iris_points,
        eye_orientation=np.radians(eye_orientation),
    )

    assert np.all(result_pupil_points == expected_pupil_points)
    assert np.all(result_iris_points == expected_iris_points)


@pytest.mark.parametrize(
    "pixel_coords,expected_intensity",
    [
        # Corners
        ((0.0, 0.0), 0.0),
        ((0.0, 1.0), 3.0),
        ((0.0, 2.0), 6.0),
        ((1.0, 0.0), 1.0),
        ((1.0, 1.0), 4.0),
        ((1.0, 2.0), 7.0),
        ((2.0, 0.0), 2.0),
        ((2.0, 1.0), 5.0),
        ((2.0, 2.0), 8.0),
        # Inside
        ((0.5, 0.5), 2),
        ((0.5, 1.5), 5),
        ((1.5, 0.5), 3),
        # Outside
        ((10.0, 0.5), 0.0),
        ((0.5, 10.0), 0.0),
        ((10.0, 10.0), 0.0),
    ],
)
def test_interpolate_pixel_intensity(pixel_coords: Tuple[float, float], expected_intensity: float) -> None:
    # fmt: off
    test_image = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]
    )
    # fmt: on

    result = interpolate_pixel_intensity(image=test_image, pixel_coords=pixel_coords)

    assert result == expected_intensity


@pytest.mark.parametrize(
    "input_img",
    [
        (np.ones(shape=(10, 10), dtype=np.uint8)),
        (np.zeros(shape=(10, 10), dtype=np.uint8)),
        (np.random.randn(100).reshape((10, 10))),
    ],
)
def test_to_uint8(input_img: np.ndarray) -> None:
    result = to_uint8(input_img)

    assert result.dtype == np.uint8
    assert np.all(result >= 0) and np.all(result <= 255)
