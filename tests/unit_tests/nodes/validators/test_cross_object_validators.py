import numpy as np
import pytest

import iris.nodes.validators.cross_object_validators as cross_obj_v
from iris.io.dataclasses import EyeCenters, GeometryPolygons, IRImage
from iris.io.errors import ExtrapolatedPolygonsInsideImageValidatorError, EyeCentersInsideImageValidatorError
from tests.unit_tests.utils import generate_arc


@pytest.mark.parametrize(
    "mock_centers,min_distance_to_border",
    [
        (EyeCenters(pupil_x=100.0, pupil_y=100.0, iris_x=100.0, iris_y=100.0), 10),
        (EyeCenters(pupil_x=10.0, pupil_y=10.0, iris_x=10.0, iris_y=10.0), 10),
        (EyeCenters(pupil_x=1430.0, pupil_y=1070.0, iris_x=1430.0, iris_y=1070.0), 10),
        (EyeCenters(pupil_x=10.0, pupil_y=100.0, iris_x=10.0, iris_y=100.0), 10),
        (EyeCenters(pupil_x=1430.0, pupil_y=100.0, iris_x=1430.0, iris_y=100.0), 10),
        (EyeCenters(pupil_x=100.0, pupil_y=10.0, iris_x=100.0, iris_y=10.0), 10),
        (EyeCenters(pupil_x=100.0, pupil_y=1070.0, iris_x=100.0, iris_y=1070.0), 10),
        (EyeCenters(pupil_x=100.0, pupil_y=100.0, iris_x=100.0, iris_y=100.0), 0),
        (EyeCenters(pupil_x=0.0, pupil_y=0.0, iris_x=0.0, iris_y=0.0), 0),
        (EyeCenters(pupil_x=1440.0, pupil_y=1080.0, iris_x=1440.0, iris_y=1080.0), 0),
        (EyeCenters(pupil_x=10.0, pupil_y=100.0, iris_x=10.0, iris_y=100.0), 0),
        (EyeCenters(pupil_x=1440.0, pupil_y=100.0, iris_x=1440.0, iris_y=100.0), 0),
        (EyeCenters(pupil_x=100.0, pupil_y=10.0, iris_x=100.0, iris_y=10.0), 0),
        (EyeCenters(pupil_x=100.0, pupil_y=1080.0, iris_x=100.0, iris_y=1080.0), 0),
    ],
    ids=[
        "simple min dist 10",
        "edge case min dist 10: upper left corner",
        "edge case min dist 10: lower right corner",
        "edge case min dist 10: left border",
        "edge case min dist 10: right border",
        "edge case min dist 10: upper border",
        "edge case min dist 10: lower border",
        "simple min dist 0",
        "edge case min dist 0: upper left corner",
        "edge case min dist 0: lower right corner",
        "edge case min dist 0: left border",
        "edge case min dist 0: right border",
        "edge case min dist 0: upper border",
        "edge case min dist 0: lower border",
    ],
)
def test_eye_centers_inside_image_validator(mock_centers: EyeCenters, min_distance_to_border: int) -> None:
    validator = cross_obj_v.EyeCentersInsideImageValidator(min_distance_to_border=min_distance_to_border)
    mock_image = IRImage(img_data=np.zeros(shape=(1080, 1440)), eye_side="right")

    try:
        validator(mock_image, mock_centers)
        assert True
    except EyeCentersInsideImageValidatorError:
        assert False, "EyeCentersInsideImageValidatorError exception raised."


@pytest.mark.parametrize(
    "mock_centers,min_distance_to_border",
    [
        (EyeCenters(pupil_x=0.0, pupil_y=0.0, iris_x=0.0, iris_y=0.0), 10),
        (EyeCenters(pupil_x=9.99, pupil_y=9.99, iris_x=9.99, iris_y=9.99), 10),
        (EyeCenters(pupil_x=1430.1, pupil_y=1070.1, iris_x=1430.1, iris_y=1070.1), 10),
        (EyeCenters(pupil_x=9.99, pupil_y=100.0, iris_x=9.99, iris_y=100.0), 10),
        (EyeCenters(pupil_x=1430.1, pupil_y=100.0, iris_x=1430.1, iris_y=100.0), 10),
        (EyeCenters(pupil_x=100.0, pupil_y=9.99, iris_x=100.0, iris_y=9.99), 10),
        (EyeCenters(pupil_x=100.0, pupil_y=1070.1, iris_x=100.0, iris_y=1070.1), 10),
        (EyeCenters(pupil_x=0.0, pupil_y=0.0, iris_x=0.0, iris_y=0.0), 10),
        (EyeCenters(pupil_x=-0.1, pupil_y=-0.1, iris_x=-0.1, iris_y=-0.1), 10),
        (EyeCenters(pupil_x=1440.1, pupil_y=1080.1, iris_x=1440.1, iris_y=1080.1), 10),
        (EyeCenters(pupil_x=-0.1, pupil_y=100.0, iris_x=-0.1, iris_y=100.0), 10),
        (EyeCenters(pupil_x=1440.1, pupil_y=100.0, iris_x=1440.1, iris_y=100.0), 10),
        (EyeCenters(pupil_x=100.0, pupil_y=-0.1, iris_x=100.0, iris_y=-0.1), 10),
        (EyeCenters(pupil_x=100.0, pupil_y=1080.1, iris_x=100.0, iris_y=1080.1), 10),
    ],
    ids=[
        "simple min dist 10",
        "edge case min dist 10: upper left corner",
        "edge case min dist 10: lower right corner",
        "edge case min dist 10: left border",
        "edge case min dist 10: right border",
        "edge case min dist 10: upper border",
        "edge case min dist 10: lower border",
        "simple min dist 0",
        "edge case min dist 0: upper left corner",
        "edge case min dist 0: lower right corner",
        "edge case min dist 0: left border",
        "edge case min dist 0: right border",
        "edge case min dist 0: upper border",
        "edge case min dist 0: lower border",
    ],
)
def test_eye_centers_inside_image_validator_raises_exception(
    mock_centers: EyeCenters, min_distance_to_border: int
) -> None:
    validator = cross_obj_v.EyeCentersInsideImageValidator(min_distance_to_border=min_distance_to_border)
    mock_image = IRImage(img_data=np.zeros(shape=(1080, 1440)), eye_side="right")

    with pytest.raises(EyeCentersInsideImageValidatorError):
        validator(mock_image, mock_centers)


@pytest.mark.parametrize(
    "mock_extrapolated_polygons",
    [
        GeometryPolygons(
            pupil_array=generate_arc(10, 720, 540, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 720, 540, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 720, 540, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(10, 100, 0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 720, 540, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 720, 540, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(10, 720, 540, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 100, 0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 720, 540, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(10, 720, 540, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 720, 540, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 100, 0, 0.0, 2 * np.pi, 360),
        ),
    ],
    ids=[
        "simple",
        "edge case: half of pupil visible",
        "edge case: half of iris visible",
        "edge case: half of eyeball visible",
    ],
)
def test_extrapolated_polygons_inside_image_validator(mock_extrapolated_polygons: GeometryPolygons) -> None:
    validator = cross_obj_v.ExtrapolatedPolygonsInsideImageValidator(
        min_pupil_allowed_percentage=0.5,
        min_iris_allowed_percentage=0.5,
        min_eyeball_allowed_percentage=0.5,
    )
    mock_image = IRImage(img_data=np.zeros(shape=(1080, 1440)), eye_side="right")

    try:
        validator(mock_image, mock_extrapolated_polygons)
        assert True
    except ExtrapolatedPolygonsInsideImageValidatorError:
        assert False, "ExtrapolatedPolygonsInsideImageValidatorError exception raised."


@pytest.mark.parametrize(
    "mock_extrapolated_polygons",
    [
        GeometryPolygons(
            pupil_array=generate_arc(10, 0, 0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 720, 540, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 720, 540, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(10, 9, 0, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 720, 540, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 720, 540, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(10, 720, 540, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 1441, 0, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 720, 540, 0.0, 2 * np.pi, 360),
        ),
        GeometryPolygons(
            pupil_array=generate_arc(10, 720, 540, 0.0, 2 * np.pi, 360),
            iris_array=generate_arc(20, 720, 540, 0.0, 2 * np.pi, 360),
            eyeball_array=generate_arc(30, 1441, 540, 0.0, 2 * np.pi, 360),
        ),
    ],
    ids=["simple", "half of pupil not visible", "half of iris not visible", "half of eyeball not visible"],
)
def test_extrapolated_polygons_inside_image_validator_raise_exception(
    mock_extrapolated_polygons: GeometryPolygons,
) -> None:
    validator = cross_obj_v.ExtrapolatedPolygonsInsideImageValidator(
        min_pupil_allowed_percentage=0.5,
        min_iris_allowed_percentage=0.5,
        min_eyeball_allowed_percentage=0.5,
    )
    mock_image = IRImage(img_data=np.zeros(shape=(1080, 1440)), eye_side="right")

    with pytest.raises(ExtrapolatedPolygonsInsideImageValidatorError):
        validator(mock_image, mock_extrapolated_polygons)
