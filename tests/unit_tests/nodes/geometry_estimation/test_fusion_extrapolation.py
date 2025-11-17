import numpy as np
import pytest

from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.nodes.geometry_estimation.fusion_extrapolation import (
    FusionExtrapolation,
    LinearExtrapolation,
    LSQEllipseFitWithRefinement,
)
from tests.unit_tests.utils import generate_arc


def _ellipse_from_arc_circle(points: np.ndarray, sx: float, sy: float) -> np.ndarray:
    cx = np.mean(points[:, 0])
    cy = np.mean(points[:, 1])
    pts = np.asarray(points, dtype=float)
    pts = (pts - np.array([cx, cy])) * np.array([sx, sy]) + np.array([cx, cy])
    return pts


@pytest.mark.parametrize(
    "input_polygons,eye_center",
    [
        (
            GeometryPolygons(
                pupil_array=generate_arc(100, 500, 500, -np.pi / 4, -3 * np.pi / 4, num_points=25000),
                iris_array=_ellipse_from_arc_circle(
                    generate_arc(300, 500, 500, np.pi - np.pi / 32, np.pi + np.pi / 32, num_points=25000),
                    sx=0.5,
                    sy=1.5,
                ),
                eyeball_array=generate_arc(500, 500, 500, 0, 2 * np.pi, num_points=25000),
            ),
            EyeCenters(pupil_x=500.0, pupil_y=500.0, iris_x=500.0, iris_y=500.0),
        ),
        (
            GeometryPolygons(
                pupil_array=generate_arc(15, 0, 0, -np.pi / 8, -3 * np.pi / 8, num_points=25000),
                iris_array=_ellipse_from_arc_circle(
                    generate_arc(40, 0, 0, np.pi * 11 / 32, np.pi * 15 / 32, num_points=25000), sx=2, sy=0.65
                ),
                eyeball_array=generate_arc(50, 0, 0, 0, 2 * np.pi, num_points=25000),
            ),
            EyeCenters(pupil_x=0.0, pupil_y=0.0, iris_x=0.0, iris_y=0.0),
        ),
    ],
    ids=["large_eye_centered", "small_eye_centered"],
)
def test_returns_circle_when_pupil_not_inside_iris(input_polygons: GeometryPolygons, eye_center: EyeCenters) -> None:
    # Make one-sided iris arc very short and elliptical so ellipse fit will cause pupil to be partially outside iris → all_inside == False

    fx = FusionExtrapolation()

    out = fx.run(input_polygons, eye_center)
    circle_poly = LinearExtrapolation(dphi=360 / 512).run(input_polygons, eye_center)
    all_inside = LSQEllipseFitWithRefinement(dphi=360 / 512)._extrapolate(input_polygons)

    assert not all_inside
    assert np.allclose(out.pupil_array, circle_poly.pupil_array)
    assert np.allclose(out.iris_array, circle_poly.iris_array)
