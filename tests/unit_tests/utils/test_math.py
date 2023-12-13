import math

import cv2
import numpy as np
import pytest

from iris.utils.math import (
    apply_weights_1d,
    area,
    cartesian2polar,
    eccentricity,
    estimate_diameter,
    orientation,
    polar2cartesian,
    polygon_length,
)
from tests.unit_tests.utils import generate_arc, generate_multiple_arcs, rotated_elliptical_contour


@pytest.mark.parametrize(
    "mock_polygon,expected_result",
    [
        (np.array([[0.0, 0.5], [0.5, 0.0], [1, 0.5], [0.5, 1]]), 0.5),
        (np.array([[0.0, 0.5], [0.5, 0.0], [0.5, 0.0], [0.5, 0.0], [1, 0.5], [0.5, 1]]), 0.5),
        (np.array([[0.0, 0.5], [0.5, 0.0], [1, 0.5], [0.5, 1], [0.0, 0.5]]), 0.5),
        (np.array([[0, 0], [1, 0], [1.5, 0.5], [1, 2], [2, 2.5], [0, 3], [0.5, 1.5]]), 3),
    ],
    ids=["square", "square + duplicate points", "square - looping", "complex polygon"],
)
def test_area(mock_polygon: np.ndarray, expected_result: float) -> None:
    result = area(mock_polygon)
    assert result == expected_result


@pytest.mark.parametrize(
    "mock_polygon",
    [
        (np.ones((100))),
        (np.ones((10, 10, 10))),
        (np.ones((10, 3))),
    ],
    ids=["1D", "3D", "2D - not points"],
)
def test_area_fails(mock_polygon: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = area(mock_polygon)


@pytest.mark.parametrize(
    "mock_polygon,expected_result",
    [
        (np.array([[0.0, 0.5], [0.5, 0.0], [0.5, 1.0], [1.0, 0.5]]), 1.0),
        (np.array([[0.0, 0.0], [0.5, -0.5], [1.0, 0.0], [0.5, 10.5]]), 11.0),
    ],
    ids=["square", "complex polygon"],
)
def test_estimate_diameter(mock_polygon: np.ndarray, expected_result: float) -> None:
    result = estimate_diameter(mock_polygon)

    assert result == expected_result


@pytest.mark.parametrize("center_x,center_y", [(0.0, 0.0), (10.0, 10.0), (-3.3, 4.4)])
def test_cartesian2polar(center_x: float, center_y: float) -> None:
    radius = 10.0

    xs = radius * np.cos(np.arange(0, 360, 1)) + center_x
    ys = radius * np.sin(np.arange(0, 360, 1)) + center_y

    rhos = np.array([radius] * 360)
    phis = np.arange(0, 2 * np.pi, np.radians(1))

    result_rhos, result_phis = cartesian2polar(xs, ys, center_x, center_y)

    np.testing.assert_almost_equal(np.sort(rhos), np.sort(result_rhos), decimal=0)
    np.testing.assert_almost_equal(np.sort(phis), np.sort(result_phis), decimal=0)


@pytest.mark.parametrize("center_x,center_y", [(0.0, 0.0), (10.0, 10.0), (-3.3, 4.4)])
def test_polar2cartesian(center_x: float, center_y: float) -> None:
    radius = 10.0
    center_x, center_y = 0.0, 0.0

    rhos = np.array([radius] * 360)
    phis = np.arange(0, 2 * np.pi, np.radians(1))

    xs = radius * np.cos(np.arange(0, 360, 1)) + center_x
    ys = radius * np.sin(np.arange(0, 360, 1)) + center_y

    result_xs, result_ys = polar2cartesian(rhos, phis, center_x, center_y)

    np.testing.assert_almost_equal(np.sort(xs), np.sort(result_xs), decimal=0)
    np.testing.assert_almost_equal(np.sort(ys), np.sort(result_ys), decimal=0)


@pytest.mark.parametrize(
    "input_contour,expected_eye_orientation",
    [
        (rotated_elliptical_contour(theta=-np.pi / 2), -np.pi / 2),
        (rotated_elliptical_contour(theta=-np.pi / 2 + 0.01), -np.pi / 2 + 0.01),
        (rotated_elliptical_contour(theta=-np.pi / 4 - 0.01), -np.pi / 4 - 0.01),
        (rotated_elliptical_contour(theta=-np.pi / 4), -np.pi / 4),
        (rotated_elliptical_contour(theta=-np.pi / 4 + 0.01), -np.pi / 4 + 0.01),
        (rotated_elliptical_contour(theta=-0.01), -0.01),
        (rotated_elliptical_contour(theta=0), 0),
        (rotated_elliptical_contour(theta=0.01), 0.01),
        (rotated_elliptical_contour(theta=np.pi / 4 - 0.01), np.pi / 4 - 0.01),
        (rotated_elliptical_contour(theta=np.pi / 4), np.pi / 4),
        (rotated_elliptical_contour(theta=np.pi / 4 + 0.01), np.pi / 4 + 0.01),
        (rotated_elliptical_contour(theta=np.pi / 2 - 0.01), np.pi / 2 - 0.01),
        (rotated_elliptical_contour(theta=np.pi / 2), -np.pi / 2),
    ],
    ids=[
        "-pi/2 => pi/2",
        "-np.pi / 2 + 0.01",
        "-np.pi / 4 - 0.01",
        "-np.pi / 4",
        "-np.pi / 4 + 0.01",
        "-0.01",
        "0",
        "0.01",
        "np.pi / 4 - 0.01",
        "np.pi / 4",
        "np.pi / 4 + 0.01",
        "np.pi / 2 - 0.01",
        "np.pi / 2",
    ],
)
def test_orientation(input_contour: np.ndarray, expected_eye_orientation: float) -> None:
    moments = cv2.moments(input_contour)
    computed_eye_orientaiton = orientation(moments)

    assert np.abs(computed_eye_orientaiton - expected_eye_orientation) < 1 / 360


@pytest.mark.parametrize(
    "input_contour,expected_eccentricity",
    [
        (rotated_elliptical_contour(a=5, b=1, theta=-np.pi / 2), 0.838),
        (rotated_elliptical_contour(a=5, b=1, theta=0), 0.838),
        (rotated_elliptical_contour(a=5, b=1, theta=0.142857), 0.838),
        (rotated_elliptical_contour(a=1, b=1, theta=0), 0),
        (rotated_elliptical_contour(a=1e20, b=1, theta=0), 0.964),
        (np.array([[0, 0], [0, 1], [1e-6, 0.5]]), 1),
        (np.array([[0, 0], [0, 1]]), 1),
    ],
    ids=[
        "Same ellipse various angles (1/3)",
        "Same ellipse various angles (2/3)",
        "Same ellipse various angles (3/3)",
        "circle",
        "almost line",
        "even more almost line",
        "perfect line",
    ],
)
def test_eccentricity(input_contour: np.ndarray, expected_eccentricity: float) -> None:
    moments = cv2.moments(input_contour)
    computed_eccentricity = eccentricity(moments)

    assert np.abs(computed_eccentricity - expected_eccentricity) < 1e-3


@pytest.mark.parametrize(
    "scores_1d,weights_1d,expected_weighted_score",
    [
        ([0, 1], [0.5, 0.5], 0.5),
        ([4, 4, 3], [1, 5, 4], 3.6),
        ([1], [0.1], 1),
        ([0, 0], [10, 1], 0),
        ([0.3, 0.21, 0.66], [0.4, 0.6, 0.11], 0.287027027027027),
    ],
)
def test_apply_weights_1d(
    scores_1d: np.ndarray,
    weights_1d: np.ndarray,
    expected_weighted_score: float,
) -> None:
    weighted_score = apply_weights_1d(scores_1d, weights_1d)
    assert np.abs(weighted_score - expected_weighted_score) < 1e-6


@pytest.mark.parametrize(
    "scores_1d,weights_1d",
    [
        ([0, 1, 1], [0.5, 0.5]),
        ([2, 3, 4, 5], [0.5, 0.5]),
        ([1, 10], [1, 2, 3]),
        ([1], [0, 2]),
    ],
)
def test_apply_weights_1d_fails(scores_1d: np.ndarray, weights_1d: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = apply_weights_1d(scores_1d, weights_1d)


@pytest.mark.parametrize(
    "mock_polygon,max_point_distance,expected_length",
    [
        (np.array([[0, 0], [0, 0], [0, 0]]), 20, 0),
        (np.array([[0, 0], [0, 1], [1, 1], [1, 0]]), 20, 4),
        (np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]]), 20, 4),
        (generate_arc(1000, 10, 30, 0, 2 * np.pi, num_points=100000), 20, 2 * np.pi * 1000),
        (
            generate_multiple_arcs(
                [
                    {
                        "radius": 1000,
                        "center_x": 0,
                        "center_y": 0,
                        "from_angle": 3 * np.pi / 4,
                        "to_angle": np.pi / 4,
                        "num_points": 25000,
                    },
                    {
                        "radius": 1000,
                        "center_x": 0,
                        "center_y": 0,
                        "from_angle": -np.pi / 4,
                        "to_angle": -3 * np.pi / 4,
                        "num_points": 25000,
                    },
                ]
            ),
            100,
            np.pi * 1000,
        ),
        (np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 4], [1, 4], [1, 5], [0, 5]]), 4, 9),
        (np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 4], [1, 4], [1, 5], [0, 5]]), 2, 6),
    ],
    ids=[
        "Empty polygon",
        "Non-looping square",
        "Looping square",
        "Large circle",
        "Two quarters of circle",
        "Two squares separated by a distance below threshold",
        "Two squares separated by a distance above threshold",
    ],
)
def test_polygon_length(mock_polygon: np.ndarray, max_point_distance: int, expected_length: float) -> None:
    computed_length = polygon_length(mock_polygon, max_point_distance=max_point_distance)
    assert math.isclose(computed_length, expected_length, rel_tol=1e-3)


@pytest.mark.parametrize(
    "mock_polygon",
    [
        (np.ones((100))),
        (np.ones((10, 10, 10))),
        (np.ones((10, 3))),
    ],
    ids=["1D", "3D", "2D - not points"],
)
def test_polygon_length_fails(mock_polygon: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = polygon_length(mock_polygon)
