from typing import Any, List, Literal

import numpy as np
import pytest
from pydantic import ValidationError

import iris.io.dataclasses as dc
from iris.io.errors import IRISPipelineError


def test_irimage_constructor() -> None:
    mock_image = np.ones(shape=(10, 10))
    mock_side = "left"

    ir_image = dc.IRImage(img_data=mock_image, eye_side=mock_side)

    assert ir_image.height == mock_image.shape[0]
    assert ir_image.width == mock_image.shape[1]


@pytest.mark.parametrize(
    "image,side",
    [
        (np.ones(shape=(10, 10)), "toto"),
        ("toto", "right"),
    ],
    ids=["wrong eye side", "not np array"],
)
def test_irimage_constructor_raises_an_exception(image: np.ndarray, side: Literal["left", "right"]) -> None:
    with pytest.raises(ValidationError):
        _ = dc.IRImage(img_data=image, eye_side=side)


def test_irimage_serialize_deserialize() -> None:
    mock_image = np.random.randint(0, 255, size=(10, 10))
    mock_side = "left"

    ir_image = dc.IRImage(img_data=mock_image, eye_side=mock_side)

    serialized_img = ir_image.serialize()
    deserialized_img = dc.IRImage.deserialize(serialized_img)

    np.testing.assert_equal(ir_image.img_data, deserialized_img.img_data)


def test_segmentation_map_constructor() -> None:
    mock_segmap = np.ones(shape=(10, 10, 2))
    mock_index2class = {0: "background", 1: "iris"}

    segmentation_map = dc.SegmentationMap(predictions=mock_segmap, index2class=mock_index2class)

    assert segmentation_map.height == mock_segmap.shape[0]
    assert segmentation_map.width == mock_segmap.shape[1]
    assert segmentation_map.nb_classes == mock_segmap.shape[2]


@pytest.mark.parametrize(
    "segmap,index2class",
    [
        ("toto", {1: "iris", 0: "background"}),
        (np.ones(shape=(10, 10, 2)), {}),
        (np.ones(shape=(10, 10, 7)), {1: "iris", 0: "background"}),
        (np.ones(shape=(10, 10)), {1: "iris", 0: "background"}),
    ],
    ids=[
        "not np array",
        "index2class not Dict[int, str]",
        "mismatch nb_classes",
        "segmap not 3-dimensional",
    ],
)
def test_segmentation_map_constructor_raises_an_exception(segmap: np.ndarray, index2class: Any) -> None:
    with pytest.raises((ValueError, ValidationError, AttributeError, IndexError)):
        _ = dc.SegmentationMap(predictions=segmap, index2class=index2class)


def test_index_of() -> None:
    segmap = dc.SegmentationMap(
        predictions=np.zeros(shape=(1440, 1080, 2)), index2class={0: "background", 1: "eyelashes"}
    )
    expected_index = 1

    result = segmap.index_of(class_name="eyelashes")

    assert result == expected_index


def test_index_of_raises_an_exception() -> None:
    segmap = dc.SegmentationMap(predictions=np.zeros(shape=(1440, 1080, 2)), index2class={0: "background", 1: "iris"})
    expected_err_msg = "Index for the `eyelashes` not found"

    with pytest.raises(ValueError) as e:
        _ = segmap.index_of(class_name="eyelashes")

    assert str(e.value) == expected_err_msg


def test_segmentation_map_serialize_deserialize() -> None:
    mock_segmap = np.random.random(size=(10, 10, 2))
    mock_index2class = {0: "background", 1: "iris"}

    segmentation_map = dc.SegmentationMap(predictions=mock_segmap, index2class=mock_index2class)

    serialized_segmap = segmentation_map.serialize()
    deserialized_segmap = dc.SegmentationMap.deserialize(serialized_segmap)

    np.testing.assert_equal(segmentation_map.predictions, deserialized_segmap.predictions)
    assert segmentation_map.index2class == deserialized_segmap.index2class


def test_geometry_polygons_constructor() -> None:
    mock_pupil_array = np.ones((40, 2))
    mock_iris_array = np.ones((150, 2))
    mock_eyeball_array = np.ones((100, 2))

    _ = dc.GeometryPolygons(pupil_array=mock_pupil_array, iris_array=mock_iris_array, eyeball_array=mock_eyeball_array)


@pytest.mark.parametrize(
    "pupil_array,iris_array,eyeball_array",
    [
        (
            np.ones((40, 3)),
            np.ones((150, 2)),
            np.ones((100, 2)),
        ),
        (
            np.ones((40, 2)),
            np.ones((150, 2)),
            None,
        ),
    ],
    ids=["input shape not 2-dimensional", "missing polygon"],
)
def test_geometry_polygons_constructor_raises_an_exception(
    pupil_array: np.ndarray, iris_array: np.ndarray, eyeball_array: np.ndarray
) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _ = dc.GeometryPolygons(pupil_array=pupil_array, iris_array=iris_array, eyeball_array=eyeball_array)


def test_geometry_polygons_serialize_deserialize() -> None:
    mock_pupil_array = np.random.random(size=(40, 2)) * 100
    mock_iris_array = np.random.random(size=(150, 2)) * 200
    mock_eyeball_array = np.random.random(size=(100, 2)) * 300

    geometry_polygons = dc.GeometryPolygons(
        pupil_array=mock_pupil_array, iris_array=mock_iris_array, eyeball_array=mock_eyeball_array
    )

    serialized_poly = geometry_polygons.serialize()
    deserialized_poly = dc.GeometryPolygons.deserialize(serialized_poly)

    np.testing.assert_equal(geometry_polygons.pupil_array, deserialized_poly.pupil_array)
    np.testing.assert_equal(geometry_polygons.iris_array, deserialized_poly.iris_array)
    np.testing.assert_equal(geometry_polygons.eyeball_array, deserialized_poly.eyeball_array)


@pytest.mark.parametrize(
    "angle",
    [(0), (1), (1.01), (-np.pi / 2), (-np.pi / 4)],
    ids=["zero", "int", "float", "-pi/2", "negative"],
)
def test_eye_orientation_constructor(angle: float) -> None:
    _ = dc.EyeOrientation(angle=angle)


@pytest.mark.parametrize(
    "angle",
    [(np.pi / 2), (-3 * np.pi / 4), (3 * np.pi / 4), ("eaux")],
    ids=["pi/2", "< -pi/2", "> pi/2", "str"],
)
def test_eye_orientation_constructor_raises_an_exception(angle: float) -> None:
    with pytest.raises((ValidationError, ValueError)):
        _ = dc.EyeOrientation(angle=angle)


def test_eye_orientation_serialize_deserialize() -> None:
    mock_angle = 0.5

    eye_orientation = dc.EyeOrientation(angle=mock_angle)

    serialized_orient = eye_orientation.serialize()
    deserialized_orient = dc.EyeOrientation.deserialize(serialized_orient)

    assert eye_orientation.angle == deserialized_orient.angle


def test_noise_mask_constructor() -> None:
    mock_mask = np.random.randint(2, size=(10, 10)).astype(bool)

    _ = dc.NoiseMask(mask=mock_mask)


@pytest.mark.parametrize(
    "noise_binary_mask",
    [
        (np.random.randint(2, size=(10, 10, 3)).astype(bool)),
        (np.random.randint(2, size=(10, 10)),),
    ],
    ids=["wrong input shape", "input not binary"],
)
def test_noise_mask_constructor_raises_an_exception(noise_binary_mask: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = dc.NoiseMask(mask=noise_binary_mask)


def test_noise_mask_serialize_deserialize() -> None:
    mock_mask = np.random.randint(2, size=(10, 10)).astype(bool)

    noise_mask = dc.NoiseMask(mask=mock_mask)

    serialized_mask = noise_mask.serialize()
    deserialized_mask = dc.NoiseMask.deserialize(serialized_mask)

    np.testing.assert_equal(noise_mask.mask, deserialized_mask.mask)


@pytest.mark.parametrize(
    "x_min,x_max,y_min,y_max",
    [
        (14, 2857.0, 142.857, 571.428),
        (0, 100.0, 0.0, 314.15),
        (-10, 10, -10, 10),
        (-20, -10, 10, 20),
        (-20, -10, -20, -10),
    ],
    ids=["regular", "min values at 0", "negative min values", "all X negative", "all negative values"],
)
def test_bounding_box(x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    _ = dc.BoundingBox(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


@pytest.mark.parametrize(
    "x_min,x_max,y_min,y_max",
    [
        (1000000, 1.0, 0.0, 10000),
        (1, 10000.0, 100000, 0.0),
        (100000, 1.0, 100000.0, 0),
        (142857, 142857.0, 142.0, 857),
        (10, 20.0, 142857.0, 142857),
    ],
    ids=[
        "x_min > x_max",
        "y_min > y_max",
        "x_min > x_max and y_min > y_max",
        "x_min = x_max",
        "y_min = y_max",
    ],
)
def test_bounding_box_constructor_raises_an_exception(x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    with pytest.raises(ValidationError):
        _ = dc.BoundingBox(x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)


def test_normalized_iris_constructor() -> None:
    mock_normalized_image = np.ones(shape=(10, 10)).astype(np.uint8)
    mock_normalized_mask = np.ones(shape=(10, 10)).astype(bool)

    _ = dc.NormalizedIris(normalized_image=mock_normalized_image, normalized_mask=mock_normalized_mask)


@pytest.mark.parametrize(
    "normalized_image,normalized_mask",
    [
        (
            np.ones(shape=(3, 10)).astype(np.uint8),
            np.ones(shape=(10, 3)).astype(bool),
        ),
        (
            np.ones(shape=(2)).astype(np.uint8),
            np.ones(shape=(2)).astype(bool),
        ),
        (
            np.ones(shape=(10, 10)).astype(np.uint8),
            np.ones(shape=(10, 10)),
        ),
        (
            np.ones(shape=(10, 10)).astype(np.float32),
            np.ones(shape=(10, 10)).astype(bool),
        ),
    ],
    ids=["resolution_mismatch", "resolution not 2D", "mask not binary", "image not uint8"],
)
def test_normalized_iris_constructor_raises_an_exception(
    normalized_image: np.ndarray, normalized_mask: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        _ = dc.NormalizedIris(normalized_image=normalized_image, normalized_mask=normalized_mask)


def test_normalized_iris_serialize_deserialize() -> None:
    mock_normalized_image = np.random.randint(0, 255, size=(10, 10)).astype(np.uint8)
    mock_normalized_mask = np.random.randint(0, 1, size=(10, 10)).astype(bool)

    normalized_iris = dc.NormalizedIris(normalized_image=mock_normalized_image, normalized_mask=mock_normalized_mask)

    serialized_normalized_iris = normalized_iris.serialize()
    deserialized_normalized_iris = dc.NormalizedIris.deserialize(serialized_normalized_iris)

    np.testing.assert_equal(normalized_iris.normalized_image, deserialized_normalized_iris.normalized_image)
    np.testing.assert_equal(normalized_iris.normalized_mask, deserialized_normalized_iris.normalized_mask)


def test_iris_filter_response_constructor() -> None:
    mock_responses = [np.random.randint(5, size=(4, 6)) for _ in range(3)]
    mock_masks = [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(3)]
    mock_iris_code_version = "v14.28"

    _ = dc.IrisFilterResponse(
        iris_responses=mock_responses, mask_responses=mock_masks, iris_code_version=mock_iris_code_version
    )


@pytest.mark.parametrize(
    "iris_responses,mask_responses,iris_code_version",
    [
        (
            [np.ones(shape=(10, 10)), "not some string", np.ones(shape=(10, 10))],
            [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(3)],
            "v3.0",
        ),
        (
            [np.random.randint(5, size=(4, 6)) for _ in range(3)],
            [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(5)],
            "v6.2",
        ),
        (
            [np.ones(shape=(10, 10)), np.ones(shape=(123, 456))],
            [np.ones(shape=(10, 10)).astype(bool), np.ones(shape=(835, 19)).astype(bool)],
            "v2.1",
        ),
        (
            [np.random.randint(5, size=(4, 6)) for _ in range(3)],
            [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(3)],
            3,
        ),
        (
            [np.random.randint(5, size=(4, 6)) for _ in range(3)],
            [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(3)],
            "not_a_version",
        ),
        (
            [np.random.randint(5, size=(4, 6)) for _ in range(3)],
            [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(3)],
            "va.b.c",
        ),
    ],
    ids=[
        "not list of np arrays",
        "iris_responses / mask_responses length difference",
        "individual iris_responses / mask_responses shape difference",
        "incorrect iris code version 1",
        "incorrect iris code version 2",
        "incorrect iris code version 3",
    ],
)
def test_iris_filter_response_constructor_raises_an_exception(
    iris_responses: List[np.ndarray], mask_responses: List[np.ndarray], iris_code_version: str
) -> None:
    with pytest.raises((ValueError, AttributeError, IRISPipelineError)):
        _ = dc.IrisFilterResponse(
            iris_responses=iris_responses, mask_responses=mask_responses, iris_code_version=iris_code_version
        )


def test_iris_filter_response_serialize_deserialize() -> None:
    mock_responses = [np.random.randint(5, size=(4, 6)) for _ in range(3)]
    mock_masks = [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(3)]
    mock_iris_code_version = "v14.28"

    iris_response = dc.IrisFilterResponse(
        iris_responses=mock_responses,
        mask_responses=mock_masks,
        iris_code_version=mock_iris_code_version,
    )

    serialized_iris_response = iris_response.serialize()
    deserialized_iris_response = dc.IrisFilterResponse.deserialize(serialized_iris_response)

    np.testing.assert_equal(iris_response.iris_responses, deserialized_iris_response.iris_responses)
    np.testing.assert_equal(iris_response.mask_responses, deserialized_iris_response.mask_responses)
    assert iris_response.iris_code_version == deserialized_iris_response.iris_code_version


def test_iris_template_constructor() -> None:
    mock_iris_codes = [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)]
    mock_mask_codes = [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)]
    mock_iris_code_version = "v14.28"

    _ = dc.IrisTemplate(
        iris_codes=mock_iris_codes,
        mask_codes=mock_mask_codes,
        iris_code_version=mock_iris_code_version,
    )


@pytest.mark.parametrize(
    "iris_codes,mask_codes,iris_code_version",
    [
        (
            [np.random.randint(2, size=(10, 10)) for _ in range(5)],
            [np.random.randint(2, size=(10, 10)) for _ in range(5)],
            "v3.0",
        ),
        (
            "not a list of arrays",
            3,
            "v3.0",
        ),
        (
            [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(3)],
            [np.random.randint(2, size=(4, 6)).astype(bool) for _ in range(5)],
            "v3.0",
        ),
        (
            [np.ones(shape=(10, 10)).astype(bool), np.ones(shape=(123, 456)).astype(bool)],
            [np.ones(shape=(10, 10)).astype(bool), np.ones(shape=(835, 19)).astype(bool)],
            "v3.0",
        ),
        (
            [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)],
            [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)],
            42,
        ),
        (
            [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)],
            [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)],
            "not_a_version",
        ),
        (
            [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)],
            [np.random.randint(2, size=(10, 10)).astype(bool) for _ in range(5)],
            "va.b.c",
        ),
    ],
    ids=[
        "not binary",
        "not array",
        "iris_codes / mask_codes length difference",
        "individual iris_codes / mask_codes shape difference",
        "incorrect iris code version 1",
        "incorrect iris code version 2",
        "incorrect iris code version 3",
    ],
)
def test_iris_template_constructor_raises_an_exception(
    iris_codes: List[np.ndarray],
    mask_codes: List[np.ndarray],
    iris_code_version: str,
) -> None:
    with pytest.raises((ValueError, ValidationError, AttributeError, IRISPipelineError)):
        _ = dc.IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version=iris_code_version)


@pytest.mark.parametrize("code_height,code_width,num_filters", [(5, 10, 2), (10, 5, 4)])
def test_iris_template_convert2old_format(code_height: int, code_width: int, num_filters: int) -> None:
    mock_iris_template = dc.IrisTemplate(
        iris_codes=[np.random.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)],
        mask_codes=[np.random.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)],
        iris_code_version="v2.1",
    )

    # 2 is for the real/complex part
    stacked_iris_codes_expected_shape = (code_height, code_width, num_filters, 2)
    stacked_mask_codes_expected_shape = (code_height, code_width, num_filters, 2)

    result_stacked_iris_codes, result_stacked_mask_codes = mock_iris_template.convert2old_format()

    assert result_stacked_iris_codes.shape == stacked_iris_codes_expected_shape
    assert result_stacked_mask_codes.shape == stacked_mask_codes_expected_shape

    for filter_idx in range(num_filters):
        np.testing.assert_equal(
            result_stacked_iris_codes[..., filter_idx, 0], mock_iris_template.iris_codes[filter_idx][..., 0]
        )
        np.testing.assert_equal(
            result_stacked_iris_codes[..., filter_idx, 1], mock_iris_template.iris_codes[filter_idx][..., 1]
        )
        np.testing.assert_equal(
            result_stacked_mask_codes[..., filter_idx, 0], mock_iris_template.mask_codes[filter_idx][..., 0]
        )
        np.testing.assert_equal(
            result_stacked_mask_codes[..., filter_idx, 1], mock_iris_template.mask_codes[filter_idx][..., 1]
        )


@pytest.mark.parametrize("code_height,code_width,num_filters", [(5, 10, 2), (10, 5, 4)])
def test_iris_template_conversion(code_height: int, code_width: int, num_filters: int) -> None:
    iris_codes = [np.random.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)]
    mask_codes = [np.random.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)]
    iris_code_version = "v2.1"

    mock_iris_template = dc.IrisTemplate(
        iris_codes=iris_codes,
        mask_codes=mask_codes,
        iris_code_version=iris_code_version,
    )

    old_format_iris_template = mock_iris_template.convert2old_format()
    new_format_iris_template = dc.IrisTemplate.convert_to_new_format(*old_format_iris_template, iris_code_version)

    np.testing.assert_equal(new_format_iris_template.iris_codes, mock_iris_template.iris_codes)
    np.testing.assert_equal(new_format_iris_template.mask_codes, mock_iris_template.mask_codes)
    assert new_format_iris_template.iris_code_version == mock_iris_template.iris_code_version


@pytest.mark.parametrize("code_height,code_width,num_filters", [(5, 10, 2), (10, 5, 4)])
def test_iris_template_conversion_reverse(code_height: int, code_width: int, num_filters: int) -> None:
    iris_codes = np.random.choice(2, size=(code_height, code_width, num_filters, 2)).astype(bool)
    mask_codes = np.random.choice(2, size=(code_height, code_width, num_filters, 2)).astype(bool)
    iris_code_version = "v2.1"

    mock_old_format_iris_template = (iris_codes, mask_codes)

    new_format_iris_template = dc.IrisTemplate.convert_to_new_format(*mock_old_format_iris_template, iris_code_version)
    old_format_iris_template = new_format_iris_template.convert2old_format()

    np.testing.assert_equal(old_format_iris_template[0], mock_old_format_iris_template[0])
    np.testing.assert_equal(old_format_iris_template[1], mock_old_format_iris_template[1])


@pytest.mark.parametrize("code_height,code_width,num_filters", [(5, 10, 2), (10, 5, 4)])
def test_iris_template_serialize(code_height: int, code_width: int, num_filters: int) -> None:
    iris_codes = [np.random.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)]
    mask_codes = [np.random.choice(2, size=(code_height, code_width, 2)).astype(bool) for _ in range(num_filters)]
    iris_code_version = "v2.1"

    mock_iris_template = dc.IrisTemplate(
        iris_codes=iris_codes,
        mask_codes=mask_codes,
        iris_code_version=iris_code_version,
    )

    serialized_iris_template = mock_iris_template.serialize()
    deserialized_iris_template = dc.IrisTemplate.deserialize(
        serialized_iris_template, array_shape=(code_height, code_width, num_filters, 2)
    )

    np.testing.assert_equal(deserialized_iris_template.iris_codes, mock_iris_template.iris_codes)
    np.testing.assert_equal(deserialized_iris_template.mask_codes, mock_iris_template.mask_codes)
    assert deserialized_iris_template.iris_code_version == mock_iris_template.iris_code_version
