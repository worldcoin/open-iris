import json
from typing import Any, List, Literal

import numpy as np
import pytest
from pydantic import ValidationError

import iris.io.dataclasses as dc
from iris.io.dataclasses import DistanceMatrix
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


class TestDistanceMatrix:
    """Test cases for the DistanceMatrix class."""

    @pytest.fixture
    def sample_distance_data(self):
        """Create sample distance data for testing."""
        return {
            (0, 1): 0.5,
            (0, 2): 0.8,
            (1, 2): 0.3,
            (0, 3): 1.2,
            (1, 3): 0.9,
            (2, 3): 0.6,
        }

    @pytest.fixture
    def distance_matrix(self, sample_distance_data):
        """Create a DistanceMatrix instance for testing."""
        return DistanceMatrix(data=sample_distance_data)

    def test_initialization(self, sample_distance_data):
        """Test DistanceMatrix initialization."""
        dm = DistanceMatrix(data=sample_distance_data)
        assert dm.data == sample_distance_data

    def test_get_existing_distance(self, distance_matrix):
        """Test getting existing distances."""
        # Test normal order
        assert distance_matrix.get(0, 1) == 0.5
        assert distance_matrix.get(1, 2) == 0.3

        # Test reverse order (should return same value)
        assert distance_matrix.get(1, 0) == 0.5
        assert distance_matrix.get(2, 1) == 0.3

    def test_get_nonexistent_distance(self, distance_matrix):
        """Test getting non-existent distances raises KeyError."""
        with pytest.raises(KeyError):
            distance_matrix.get(0, 4)  # (0, 4) doesn't exist in the data

    def test_to_numpy(self, distance_matrix):
        """Test conversion to numpy array."""
        numpy_matrix = distance_matrix.to_numpy()

        # Check shape (should be 4x4 based on our sample data)
        assert numpy_matrix.shape == (4, 4)

        # Check specific values
        assert numpy_matrix[0, 1] == 0.5
        assert numpy_matrix[1, 0] == 0.5  # symmetry
        assert numpy_matrix[0, 2] == 0.8
        assert numpy_matrix[2, 0] == 0.8  # symmetry
        assert numpy_matrix[1, 2] == 0.3
        assert numpy_matrix[2, 1] == 0.3  # symmetry

        # Check diagonal (should be zeros)
        assert numpy_matrix[0, 0] == 0.0
        assert numpy_matrix[1, 1] == 0.0
        assert numpy_matrix[2, 2] == 0.0
        assert numpy_matrix[3, 3] == 0.0

        # Check that it's symmetric
        assert np.allclose(numpy_matrix, numpy_matrix.T)

    def test_to_matrix(self, distance_matrix):
        """Test to_matrix method (should be same as to_numpy)."""
        matrix_result = distance_matrix.to_matrix()
        numpy_result = distance_matrix.to_numpy()

        assert np.array_equal(matrix_result, numpy_result)

    def test_nb_templates(self, distance_matrix):
        """Test nb_templates property."""
        # Our sample data has templates with indices 0, 1, 2, 3
        assert distance_matrix.nb_templates == 4

    def test_nb_templates_with_gaps(self):
        """Test nb_templates with non-consecutive indices."""
        data = {(0, 5): 0.5, (2, 7): 0.8, (5, 7): 0.3}
        dm = DistanceMatrix(data=data)

        # Should count unique indices: 0, 2, 5, 7
        assert dm.nb_templates == 4

    def test_serialize(self, distance_matrix, sample_distance_data):
        """Test serialization."""
        serialized = distance_matrix.serialize()
        expected = {f"{i}_{j}": v for (i, j), v in sample_distance_data.items()}
        assert serialized == expected

    def test_deserialize(self, sample_distance_data):
        """Test deserialization."""
        input_data = {f"{i}_{j}": v for (i, j), v in sample_distance_data.items()}
        deserialized = DistanceMatrix.deserialize(input_data)
        assert isinstance(deserialized, DistanceMatrix)
        assert deserialized.data == sample_distance_data

    def test_deserialize_invalid_key_format_missing_separator(self):
        """DistanceMatrix.deserialize should raise ValueError when key lacks underscore separator."""
        bad_input = {"0-1": 0.5}
        with pytest.raises(ValueError, match="Invalid distance matrix key format"):
            DistanceMatrix.deserialize(bad_input)

    def test_deserialize_invalid_key_format_non_integer_indices(self):
        """DistanceMatrix.deserialize should raise ValueError when indices are not integers."""
        bad_input = {"1_a": 0.7}
        with pytest.raises(ValueError, match="Invalid distance matrix key format"):
            DistanceMatrix.deserialize(bad_input)

    def test_serialize_deserialize_roundtrip(self, distance_matrix):
        """Test that serialize followed by deserialize returns equivalent object."""
        serialized = distance_matrix.serialize()
        deserialized = DistanceMatrix.deserialize(serialized)

        assert deserialized.data == distance_matrix.data

    def test_empty_distance_matrix(self):
        """Test DistanceMatrix with empty data."""
        empty_dm = DistanceMatrix(data={})

        assert empty_dm.nb_templates == 0
        assert empty_dm.serialize() == {}

        # to_numpy should return empty array
        numpy_matrix = empty_dm.to_numpy()
        assert numpy_matrix.shape == (0, 0)

    def test_single_distance(self):
        """Test DistanceMatrix with only one distance."""
        data = {(0, 1): 0.5}
        dm = DistanceMatrix(data=data)

        assert dm.nb_templates == 2
        assert dm.get(0, 1) == 0.5
        assert dm.get(1, 0) == 0.5

        numpy_matrix = dm.to_numpy()
        assert numpy_matrix.shape == (2, 2)
        assert numpy_matrix[0, 1] == 0.5
        assert numpy_matrix[1, 0] == 0.5

    def test_immutability(self, distance_matrix):
        """Test that DistanceMatrix is immutable (inherits from ImmutableModel)."""
        # Should not be able to modify the data attribute
        with pytest.raises(Exception):
            distance_matrix.data = {}

    def test_to_numpy_with_large_matrix(self):
        """Test to_numpy with a larger matrix."""
        # Create data for a 5x5 matrix
        data = {}
        for i in range(5):
            for j in range(i + 1, 5):
                data[(i, j)] = float(i + j)

        dm = DistanceMatrix(data=data)
        numpy_matrix = dm.to_numpy()

        assert numpy_matrix.shape == (5, 5)
        assert np.allclose(numpy_matrix, numpy_matrix.T)  # symmetric

        # Check some specific values
        assert numpy_matrix[0, 1] == 1.0
        assert numpy_matrix[1, 0] == 1.0
        assert numpy_matrix[2, 4] == 6.0
        assert numpy_matrix[4, 2] == 6.0

    def test_len_method(self, distance_matrix):
        """Test that __len__ returns the correct number of distances."""
        assert len(distance_matrix) == 6  # Our sample data has 6 distances

    def test_len_method_empty_matrix(self):
        """Test that __len__ works correctly with empty distance matrix."""
        empty_dm = DistanceMatrix(data={})
        assert len(empty_dm) == 0

    def test_len_method_single_distance(self):
        """Test that __len__ works correctly with a single distance."""
        data = {(0, 1): 0.5}
        dm = DistanceMatrix(data=data)
        assert len(dm) == 1

    def test_len_method_large_matrix(self):
        """Test that __len__ works correctly with a larger matrix."""
        # Create data for a 5x5 matrix (10 distances)
        data = {}
        for i in range(5):
            for j in range(i + 1, 5):
                data[(i, j)] = float(i + j)

        dm = DistanceMatrix(data=data)
        assert len(dm) == 10  # 5 choose 2 = 10 distances

    def test_len_method_consistency(self, distance_matrix):
        """Test that __len__ is consistent with the data dictionary length."""
        assert len(distance_matrix) == len(distance_matrix.data)

    def test_len_method_vs_nb_templates(self, distance_matrix):
        """Test the relationship between __len__ and nb_templates."""
        # For n templates, we expect n*(n-1)/2 distances
        n = distance_matrix.nb_templates
        expected_distances = n * (n - 1) // 2
        assert len(distance_matrix) == expected_distances


class TestAlignedTemplates:
    """Test cases for the AlignedTemplates class."""

    @pytest.fixture
    def sample_iris_templates(self, request):
        """Create sample IrisTemplate instances for testing.

        Can be parametrized with 'nb_templates' marker to specify number of templates.
        Default is 3 templates.
        """
        # Get number of templates from marker, default to 3
        nb_templates = getattr(request.node.get_closest_marker("nb_templates"), "args", [3])[0]

        iris_codes = [
            np.random.choice(2, size=(16, 200, 2)).astype(bool),  # Realistic size
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
        ]
        mask_codes = [
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),  # Mostly valid
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
        ]
        iris_code_version = "v2.1"

        return [
            dc.IrisTemplate(
                iris_codes=iris_codes,
                mask_codes=mask_codes,
                iris_code_version=iris_code_version,
            )
            for _ in range(nb_templates)
        ]

    @pytest.fixture
    def sample_distance_matrix(self):
        """Create sample DistanceMatrix for testing."""
        distance_data = {
            (0, 1): 0.5,
            (0, 2): 0.8,
            (1, 2): 0.3,
        }
        return DistanceMatrix(data=distance_data)

    @pytest.fixture
    def aligned_templates(self, sample_iris_templates, sample_distance_matrix):
        """Create an AlignedTemplates instance for testing."""
        return dc.AlignedTemplates(
            templates=sample_iris_templates, distances=sample_distance_matrix, reference_template_id=0
        )

    def test_initialization(self, sample_iris_templates, sample_distance_matrix):
        """Test AlignedTemplates initialization."""
        aligned_templates = dc.AlignedTemplates(
            templates=sample_iris_templates, distances=sample_distance_matrix, reference_template_id=1
        )

        assert aligned_templates.templates == sample_iris_templates
        assert aligned_templates.distances == sample_distance_matrix
        assert aligned_templates.reference_template_id == 1

    def test_reference_template_property(self, aligned_templates, sample_iris_templates):
        """Test reference_template property."""
        reference_template = aligned_templates.reference_template
        assert reference_template == sample_iris_templates[0]  # reference_template_id is 0

    def test_reference_template_property_different_id(self, sample_iris_templates, sample_distance_matrix):
        """Test reference_template property with different reference ID."""
        aligned_templates = dc.AlignedTemplates(
            templates=sample_iris_templates, distances=sample_distance_matrix, reference_template_id=2
        )

        reference_template = aligned_templates.reference_template
        assert reference_template == sample_iris_templates[2]

    def test_get_distance(self, aligned_templates):
        """Test get_distance method."""
        # Test existing distances
        assert aligned_templates.get_distance(0, 1) == 0.5
        assert aligned_templates.get_distance(1, 0) == 0.5  # symmetry
        assert aligned_templates.get_distance(0, 2) == 0.8
        assert aligned_templates.get_distance(2, 0) == 0.8  # symmetry
        assert aligned_templates.get_distance(1, 2) == 0.3
        assert aligned_templates.get_distance(2, 1) == 0.3  # symmetry

    def test_get_distance_nonexistent(self, aligned_templates):
        """Test get_distance method with non-existent distance raises KeyError."""
        with pytest.raises(KeyError):
            aligned_templates.get_distance(0, 3)  # (0, 3) doesn't exist in the distance matrix

    def test_serialize(self, aligned_templates):
        """Test serialization."""
        serialized = aligned_templates.serialize()

        # Check that all expected keys are present
        assert "templates" in serialized
        assert "distances" in serialized
        assert "reference_template_id" in serialized

        # Check values
        assert serialized["reference_template_id"] == 0
        assert len(serialized["templates"]) == 3
        assert isinstance(serialized["distances"], dict)
        assert isinstance(serialized["templates"][0]["iris_codes"], str)

    def test_deserialize(self, sample_iris_templates, sample_distance_matrix):
        """Test deserialization."""
        # Create serialized data
        serialized_data = {
            "templates": [template.serialize() for template in sample_iris_templates],
            "distances": sample_distance_matrix.serialize(),
            "reference_template_id": 1,
        }

        deserialized = dc.AlignedTemplates.deserialize(serialized_data, array_shape=(16, 200, 2, 2))

        assert isinstance(deserialized, dc.AlignedTemplates)
        assert deserialized.reference_template_id == 1
        assert len(deserialized.templates) == 3
        assert deserialized.distances.nb_templates == 3

    def test_serialize_deserialize_roundtrip(self, aligned_templates):
        """Test that serialize followed by deserialize returns equivalent object."""
        serialized = aligned_templates.serialize()
        deserialized = dc.AlignedTemplates.deserialize(serialized, array_shape=(16, 200, 2, 2))

        # Check that the deserialized object has the same properties
        assert deserialized.reference_template_id == aligned_templates.reference_template_id
        assert len(deserialized.templates) == len(aligned_templates.templates)
        assert deserialized.distances.nb_templates == aligned_templates.distances.nb_templates

    def test_empty_templates_list(self):
        """Test AlignedTemplates with empty templates list."""
        empty_distance_matrix = DistanceMatrix(data={})

        with pytest.raises(ValueError):
            dc.AlignedTemplates(templates=[], distances=empty_distance_matrix, reference_template_id=0)

    def test_invalid_reference_template_id(self, sample_iris_templates, sample_distance_matrix):
        """Test AlignedTemplates with invalid reference template ID."""
        with pytest.raises(ValueError):
            dc.AlignedTemplates(
                templates=sample_iris_templates,
                distances=sample_distance_matrix,
                reference_template_id=5,  # Index out of range
            )

    def test_negative_reference_template_id(self, sample_iris_templates, sample_distance_matrix):
        """Test AlignedTemplates with negative reference template ID."""
        with pytest.raises(ValueError):
            dc.AlignedTemplates(
                templates=sample_iris_templates, distances=sample_distance_matrix, reference_template_id=-1
            )

    def test_immutability(self, aligned_templates):
        """Test that AlignedTemplates is immutable (inherits from ImmutableModel)."""
        # Should not be able to modify the attributes
        with pytest.raises(Exception):
            aligned_templates.reference_template_id = 1

    def test_distance_matrix_integration(self, aligned_templates):
        """Test integration with DistanceMatrix methods."""
        # Test that we can access distance matrix properties
        assert aligned_templates.distances.nb_templates == 3

        # Test that we can convert to numpy
        numpy_matrix = aligned_templates.distances.to_numpy()
        assert numpy_matrix.shape == (3, 3)
        assert np.allclose(numpy_matrix, numpy_matrix.T)  # symmetric

    def test_multiple_reference_templates(self, sample_iris_templates, sample_distance_matrix):
        """Test with different reference template IDs."""
        for ref_id in range(len(sample_iris_templates)):
            aligned_templates = dc.AlignedTemplates(
                templates=sample_iris_templates, distances=sample_distance_matrix, reference_template_id=ref_id
            )

            reference_template = aligned_templates.reference_template
            assert reference_template == sample_iris_templates[ref_id]

    def test_large_number_of_templates(self):
        """Test AlignedTemplates with a larger number of templates."""
        # Create 5 templates
        iris_codes = [np.random.choice(2, size=(8, 8)).astype(bool) for _ in range(2)]
        mask_codes = [np.random.choice(2, size=(8, 8)).astype(bool) for _ in range(2)]
        iris_code_version = "v14.28"

        templates = [
            dc.IrisTemplate(
                iris_codes=iris_codes,
                mask_codes=mask_codes,
                iris_code_version=iris_code_version,
            )
            for _ in range(5)
        ]

        # Create distance matrix for 5 templates
        distance_data = {}
        for i in range(5):
            for j in range(i + 1, 5):
                distance_data[(i, j)] = float(i + j) / 10.0

        distance_matrix = DistanceMatrix(data=distance_data)

        aligned_templates = dc.AlignedTemplates(templates=templates, distances=distance_matrix, reference_template_id=2)

        assert len(aligned_templates.templates) == 5
        assert aligned_templates.distances.nb_templates == 5
        assert aligned_templates.reference_template == templates[2]

        # Test some distances
        assert aligned_templates.get_distance(0, 1) == 0.1
        assert aligned_templates.get_distance(2, 4) == 0.6

    def test_distances_match_templates_validator(self, sample_iris_templates):
        """Test that the validator correctly checks distances match templates."""
        # Create a distance matrix with wrong number of templates
        distance_data = {
            (0, 1): 0.5,
            (0, 2): 0.8,
            (1, 2): 0.3,
            (0, 3): 1.2,  # This adds a 4th template index
        }
        distance_matrix = DistanceMatrix(data=distance_data)

        # Should raise ValueError because we have 3 templates but 4 template indices in distances
        with pytest.raises(ValueError, match="Number of templates \\(3\\) does not match number of distances \\(4\\)"):
            dc.AlignedTemplates(
                templates=sample_iris_templates,  # 3 templates
                distances=distance_matrix,  # 4 template indices (0, 1, 2, 3)
                reference_template_id=0,
            )

    def test_distances_match_templates_validator_insufficient_distances(self, sample_iris_templates):
        """Test that the validator correctly checks when there are insufficient distances."""
        # Create a distance matrix with fewer template indices than templates
        distance_data = {
            (0, 1): 0.5,  # Only covers templates 0 and 1
        }
        distance_matrix = DistanceMatrix(data=distance_data)

        # Should raise ValueError because we have 3 templates but only 2 template indices in distances
        with pytest.raises(ValueError, match="Number of templates \\(3\\) does not match number of distances \\(2\\)"):
            dc.AlignedTemplates(
                templates=sample_iris_templates,  # 3 templates
                distances=distance_matrix,  # 2 template indices (0, 1)
                reference_template_id=0,
            )

    def test_distances_match_templates_validator_correct(self, sample_iris_templates, sample_distance_matrix):
        """Test that the validator passes when distances match templates."""
        # This should work correctly
        aligned_templates = dc.AlignedTemplates(
            templates=sample_iris_templates,  # 3 templates
            distances=sample_distance_matrix,  # 3 template indices (0, 1, 2)
            reference_template_id=0,
        )

        # Verify it was created successfully
        assert len(aligned_templates.templates) == 3
        assert aligned_templates.distances.nb_templates == 3

    def test_reference_template_id_validator_negative(self, sample_iris_templates, sample_distance_matrix):
        """Test that the validator correctly checks for negative reference_template_id."""
        # Should raise ValueError because reference_template_id is negative
        with pytest.raises(ValueError, match="reference_template_id \\(-1\\) cannot be negative"):
            dc.AlignedTemplates(
                templates=sample_iris_templates,  # 3 templates
                distances=sample_distance_matrix,
                reference_template_id=-1,
            )

    def test_reference_template_id_validator_out_of_range(self, sample_iris_templates, sample_distance_matrix):
        """Test that the validator correctly checks for out-of-range reference_template_id."""
        # Should raise ValueError because reference_template_id is out of range
        with pytest.raises(ValueError, match="reference_template_id \\(5\\) is out of range"):
            dc.AlignedTemplates(
                templates=sample_iris_templates,  # 3 templates (indices 0, 1, 2)
                distances=sample_distance_matrix,
                reference_template_id=5,  # Out of range
            )

    def test_reference_template_id_validator_boundary_values(self, sample_iris_templates, sample_distance_matrix):
        """Test that the validator accepts valid boundary values."""
        # Test with valid boundary values (0 and 2 for 3 templates)
        aligned_templates_0 = dc.AlignedTemplates(
            templates=sample_iris_templates,
            distances=sample_distance_matrix,
            reference_template_id=0,  # Valid: first template
        )
        assert aligned_templates_0.reference_template_id == 0

        aligned_templates_2 = dc.AlignedTemplates(
            templates=sample_iris_templates,
            distances=sample_distance_matrix,
            reference_template_id=2,  # Valid: last template
        )
        assert aligned_templates_2.reference_template_id == 2

    def test_reference_template_id_validator_single_template(self):
        """Test that the validator works correctly with a single template."""
        # Create a single template
        iris_codes = [np.random.choice(2, size=(16, 200, 2)).astype(bool)]
        mask_codes = [np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool)]
        iris_code_version = "v2.1"

        template = dc.IrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            iris_code_version=iris_code_version,
        )

        # Create distance matrix for single template (empty since no distances needed)
        distance_matrix = DistanceMatrix(data={})

        # Should work with reference_template_id = 0
        aligned_templates = dc.AlignedTemplates(
            templates=[template], distances=distance_matrix, reference_template_id=0
        )
        assert aligned_templates.reference_template_id == 0

        # Should fail with reference_template_id = 1
        with pytest.raises(ValueError, match="reference_template_id \\(1\\) is out of range"):
            dc.AlignedTemplates(templates=[template], distances=distance_matrix, reference_template_id=1)

    def test_len_method(self, aligned_templates):
        """Test that __len__ returns the correct number of templates."""
        assert len(aligned_templates) == 3

    def test_len_method_single_template(self):
        """Test that __len__ works correctly with a single template."""
        # Create a single template
        iris_codes = [np.random.choice(2, size=(16, 200, 2)).astype(bool)]
        mask_codes = [np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool)]
        iris_code_version = "v2.1"

        template = dc.IrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            iris_code_version=iris_code_version,
        )

        # Create distance matrix for single template (empty since no distances needed)
        distance_matrix = DistanceMatrix(data={})

        aligned_templates = dc.AlignedTemplates(
            templates=[template], distances=distance_matrix, reference_template_id=0
        )

        assert len(aligned_templates) == 1

    def test_len_method_multiple_templates(self):
        """Test that __len__ works correctly with multiple templates."""
        # Create 5 templates
        iris_codes = [np.random.choice(2, size=(8, 8)).astype(bool) for _ in range(2)]
        mask_codes = [np.random.choice(2, size=(8, 8)).astype(bool) for _ in range(2)]
        iris_code_version = "v14.28"

        templates = [
            dc.IrisTemplate(
                iris_codes=iris_codes,
                mask_codes=mask_codes,
                iris_code_version=iris_code_version,
            )
            for _ in range(5)
        ]

        # Create distance matrix for 5 templates
        distance_data = {}
        for i in range(5):
            for j in range(i + 1, 5):
                distance_data[(i, j)] = float(i + j) / 10.0

        distance_matrix = DistanceMatrix(data=distance_data)

        aligned_templates = dc.AlignedTemplates(templates=templates, distances=distance_matrix, reference_template_id=2)

        assert len(aligned_templates) == 5

    def test_len_method_consistency(self, aligned_templates):
        """Test that __len__ is consistent with other length-related properties."""
        assert len(aligned_templates) == len(aligned_templates.templates)
        assert len(aligned_templates) == aligned_templates.distances.nb_templates


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda: dc.EyeOrientation(angle=0.0), id="EyeOrientation"),
        pytest.param(lambda: dc.EyeCenters(pupil_x=0.0, pupil_y=0.0, iris_x=1.0, iris_y=1.0), id="EyeCenters"),
        pytest.param(lambda: dc.Offgaze(score=0.5), id="Offgaze"),
        pytest.param(lambda: dc.Sharpness(score=0.1), id="Sharpness"),
        pytest.param(
            lambda: dc.PupilToIrisProperty(pupil_to_iris_diameter_ratio=0.5, pupil_to_iris_center_dist_ratio=0.2),
            id="PupilToIrisProperty",
        ),
        pytest.param(
            lambda: dc.Landmarks(
                pupil_landmarks=np.ones((2, 2), dtype=float),
                iris_landmarks=np.ones((2, 2), dtype=float),
                eyeball_landmarks=np.ones((2, 2), dtype=float),
            ),
            id="Landmarks",
        ),
        pytest.param(lambda: dc.BoundingBox(x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0), id="BoundingBox"),
        pytest.param(
            lambda: dc.IrisTemplate(
                iris_codes=[np.ones((2, 2, 2), dtype=bool), np.ones((2, 2, 2), dtype=bool)],
                mask_codes=[np.ones((2, 2, 2), dtype=bool), np.ones((2, 2, 2), dtype=bool)],
                iris_code_version="v2.1",
            ),
            id="IrisTemplate",
        ),
        pytest.param(
            lambda: dc.WeightedIrisTemplate(
                iris_codes=[np.ones((2, 2, 2), dtype=bool), np.ones((2, 2, 2), dtype=bool)],
                mask_codes=[np.ones((2, 2, 2), dtype=bool), np.ones((2, 2, 2), dtype=bool)],
                weights=[np.ones((2, 2, 2), dtype=np.float32), np.ones((2, 2, 2), dtype=np.float32)],
                iris_code_version="v2.1",
            ),
            id="WeightedIrisTemplate",
        ),
        pytest.param(lambda: dc.EyeOcclusion(visible_fraction=0.5), id="EyeOcclusion"),
        pytest.param(
            lambda: dc.DistanceMatrix(data={(0, 1): 0.1, (0, 2): 0.2, (1, 2): 0.3}),
            id="DistanceMatrix",
        ),
        pytest.param(
            lambda: dc.AlignedTemplates(
                templates=[
                    dc.IrisTemplate(
                        iris_codes=[np.ones((2, 2, 2), dtype=bool)],
                        mask_codes=[np.ones((2, 2, 2), dtype=bool)],
                        iris_code_version="v2.1",
                    ),
                    dc.IrisTemplate(
                        iris_codes=[np.ones((2, 2, 2), dtype=bool)],
                        mask_codes=[np.ones((2, 2, 2), dtype=bool)],
                        iris_code_version="v2.1",
                    ),
                ],
                distances=DistanceMatrix(data={(0, 1): 0.5}),
                reference_template_id=0,
            ),
            id="AlignedTemplates",
        ),
    ],
)
def test_all_metadata_dataclasses_are_json_serializable(factory) -> None:
    """Ensure that serialize() output for all dataclasses can be JSON-serialized.

    Uses a NumPy-aware default to convert arrays and scalars.
    """
    obj = factory()
    serialized = obj.serialize()
    # Should not raise
    json.dumps(serialized)


class TestWeightedIrisTemplate:
    """Test cases for the WeightedIrisTemplate class."""

    @pytest.fixture
    def sample_weighted_iris_template_data(self):
        """Create sample data for WeightedIrisTemplate testing."""
        iris_codes = [
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
        ]
        mask_codes = [
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
        ]
        weights = [
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
        ]
        iris_code_version = "v2.1"

        return {
            "iris_codes": iris_codes,
            "mask_codes": mask_codes,
            "weights": weights,
            "iris_code_version": iris_code_version,
        }

    @pytest.fixture
    def weighted_iris_template(self, sample_weighted_iris_template_data):
        """Create a WeightedIrisTemplate instance for testing."""
        return dc.WeightedIrisTemplate(**sample_weighted_iris_template_data)

    def test_initialization(self, sample_weighted_iris_template_data):
        """Test WeightedIrisTemplate initialization."""
        weighted_template = dc.WeightedIrisTemplate(**sample_weighted_iris_template_data)

        assert len(weighted_template.iris_codes) == 2
        assert len(weighted_template.mask_codes) == 2
        assert len(weighted_template.weights) == 2
        assert weighted_template.iris_code_version == "v2.1"

        # Check that weights have correct shape and type
        for i, weight in enumerate(weighted_template.weights):
            assert weight.shape == weighted_template.iris_codes[i].shape
            assert weight.dtype == np.float32
            assert np.all(weight >= 0)

    def test_weights_length_mismatch(self, sample_weighted_iris_template_data):
        """Test that initialization fails when weights and iris_codes have different lengths."""
        # Create weights with wrong length
        wrong_weights = [sample_weighted_iris_template_data["weights"][0]]  # Only one weight instead of two

        with pytest.raises(ValueError, match="weights and iris_codes must have same length"):
            dc.WeightedIrisTemplate(
                iris_codes=sample_weighted_iris_template_data["iris_codes"],
                mask_codes=sample_weighted_iris_template_data["mask_codes"],
                weights=wrong_weights,
                iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
            )

    def test_weights_shape_mismatch(self, sample_weighted_iris_template_data):
        """Test that initialization fails when weights and iris_codes have different shapes."""
        # Create weights with wrong shape
        wrong_weights = [
            np.random.uniform(0.0, 1.0, size=(16, 100, 2)).astype(np.float32),  # Wrong width
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
        ]

        with pytest.raises(ValueError, match="Shape mismatch at wavelet 0"):
            dc.WeightedIrisTemplate(
                iris_codes=sample_weighted_iris_template_data["iris_codes"],
                mask_codes=sample_weighted_iris_template_data["mask_codes"],
                weights=wrong_weights,
                iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
            )

    def test_negative_weights(self, sample_weighted_iris_template_data):
        """Test that initialization fails when weights contain negative values."""
        # Create weights with negative values
        negative_weights = [
            np.random.uniform(-1.0, 1.0, size=(16, 200, 2)).astype(np.float32),  # Contains negative values
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
        ]

        with pytest.raises(ValueError, match="All weights must be >= 0"):
            dc.WeightedIrisTemplate(
                iris_codes=sample_weighted_iris_template_data["iris_codes"],
                mask_codes=sample_weighted_iris_template_data["mask_codes"],
                weights=negative_weights,
                iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
            )

    def test_non_float_weights(self, sample_weighted_iris_template_data):
        """Test that initialization fails when weights are not float arrays."""
        # Create weights with wrong dtype
        int_weights = [
            np.random.randint(0, 2, size=(16, 200, 2)).astype(np.int32),  # Integer instead of float
            np.random.randint(0, 2, size=(16, 200, 2)).astype(np.int32),
        ]

        with pytest.raises(ValidationError, match="Weight must be float array"):
            dc.WeightedIrisTemplate(
                iris_codes=sample_weighted_iris_template_data["iris_codes"],
                mask_codes=sample_weighted_iris_template_data["mask_codes"],
                weights=int_weights,
                iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
            )

    def test_missing_iris_codes(self, sample_weighted_iris_template_data):
        """Test that initialization fails when iris_codes is missing."""
        with pytest.raises(ValueError, match="iris_codes and weights must both be provided"):
            dc.WeightedIrisTemplate(
                mask_codes=sample_weighted_iris_template_data["mask_codes"],
                weights=sample_weighted_iris_template_data["weights"],
                iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
            )

    def test_missing_weights(self, sample_weighted_iris_template_data):
        """Test that initialization fails when weights is missing."""
        with pytest.raises(ValueError, match="iris_codes and weights must both be provided"):
            dc.WeightedIrisTemplate(
                iris_codes=sample_weighted_iris_template_data["iris_codes"],
                mask_codes=sample_weighted_iris_template_data["mask_codes"],
                iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
            )

    def test_serialize(self, weighted_iris_template):
        """Test serialization of WeightedIrisTemplate."""
        serialized = weighted_iris_template.serialize()

        # Check that all expected keys are present
        assert "iris_codes" in serialized
        assert "mask_codes" in serialized
        assert "weights" in serialized
        assert "iris_code_version" in serialized

        # Check that values are strings (base64 encoded)
        assert isinstance(serialized["iris_codes"], str)
        assert isinstance(serialized["mask_codes"], str)
        assert isinstance(serialized["weights"], str)
        assert serialized["iris_code_version"] == "v2.1"

    def test_deserialize(self, sample_weighted_iris_template_data):
        """Test deserialization of WeightedIrisTemplate."""
        # Create a template first
        original_template = dc.WeightedIrisTemplate(**sample_weighted_iris_template_data)

        # Serialize and then deserialize
        serialized = original_template.serialize()
        deserialized = dc.WeightedIrisTemplate.deserialize(serialized, array_shape=(16, 200, 2, 2))

        # Check that the deserialized object matches the original
        assert len(deserialized.iris_codes) == len(original_template.iris_codes)
        assert len(deserialized.mask_codes) == len(original_template.mask_codes)
        assert len(deserialized.weights) == len(original_template.weights)
        assert deserialized.iris_code_version == original_template.iris_code_version

        # Check that the data is preserved
        for i in range(len(original_template.iris_codes)):
            np.testing.assert_array_equal(deserialized.iris_codes[i], original_template.iris_codes[i])
            np.testing.assert_array_equal(deserialized.mask_codes[i], original_template.mask_codes[i])
            np.testing.assert_array_almost_equal(deserialized.weights[i], original_template.weights[i])

    def test_serialize_deserialize_roundtrip(self, weighted_iris_template):
        """Test that serialize followed by deserialize returns equivalent object."""
        serialized = weighted_iris_template.serialize()
        deserialized = dc.WeightedIrisTemplate.deserialize(serialized, array_shape=(16, 200, 2, 2))

        # Check that the deserialized object has the same properties
        assert deserialized.iris_code_version == weighted_iris_template.iris_code_version
        assert len(deserialized.iris_codes) == len(weighted_iris_template.iris_codes)
        assert len(deserialized.mask_codes) == len(weighted_iris_template.mask_codes)
        assert len(deserialized.weights) == len(weighted_iris_template.weights)

    def test_convert2old_format(self, weighted_iris_template):
        """Test convert2old_format method."""
        old_iris_codes, old_mask_codes, old_weights = weighted_iris_template.convert2old_format()

        # Check shapes
        assert old_iris_codes.shape == (16, 200, 2, 2)  # (height, width, nb_wavelets, 2)
        assert old_mask_codes.shape == (16, 200, 2, 2)
        assert old_weights.shape == (16, 200, 2, 2)

        # Check that the conversion is correct
        for i in range(2):  # 2 wavelets
            np.testing.assert_array_equal(old_iris_codes[:, :, i, :], weighted_iris_template.iris_codes[i])
            np.testing.assert_array_equal(old_mask_codes[:, :, i, :], weighted_iris_template.mask_codes[i])
            np.testing.assert_array_almost_equal(old_weights[:, :, i, :], weighted_iris_template.weights[i])

    def test_convert_to_new_format(self, sample_weighted_iris_template_data):
        """Test convert_to_new_format static method."""
        # Create old format data
        old_iris_codes = np.random.choice(2, size=(16, 200, 2, 2)).astype(bool)
        old_mask_codes = np.random.choice(2, size=(16, 200, 2, 2)).astype(bool)
        old_weights = np.random.uniform(0.0, 1.0, size=(16, 200, 2, 2)).astype(np.float32)
        iris_code_version = "v2.1"

        # Convert to new format
        new_template = dc.WeightedIrisTemplate.convert_to_new_format(
            iris_codes=old_iris_codes,
            mask_codes=old_mask_codes,
            weights=old_weights,
            iris_code_version=iris_code_version,
        )

        # Check that the conversion is correct
        assert len(new_template.iris_codes) == 2
        assert len(new_template.mask_codes) == 2
        assert len(new_template.weights) == 2
        assert new_template.iris_code_version == iris_code_version

        # Check that the data is preserved
        for i in range(2):
            np.testing.assert_array_equal(new_template.iris_codes[i], old_iris_codes[:, :, i, :])
            np.testing.assert_array_equal(new_template.mask_codes[i], old_mask_codes[:, :, i, :])
            np.testing.assert_array_almost_equal(new_template.weights[i], old_weights[:, :, i, :])

    def test_inheritance_from_iris_template(self, weighted_iris_template):
        """Test that WeightedIrisTemplate inherits correctly from IrisTemplate."""
        # Check that it has all the properties of IrisTemplate
        assert hasattr(weighted_iris_template, "iris_codes")
        assert hasattr(weighted_iris_template, "mask_codes")
        assert hasattr(weighted_iris_template, "iris_code_version")
        assert hasattr(weighted_iris_template, "serialize")
        assert hasattr(weighted_iris_template, "deserialize")

        # Check that it's an instance of both classes
        assert isinstance(weighted_iris_template, dc.WeightedIrisTemplate)
        assert isinstance(weighted_iris_template, dc.IrisTemplate)

    def test_weights_property_access(self, weighted_iris_template):
        """Test that weights property can be accessed correctly."""
        weights = weighted_iris_template.weights

        assert len(weights) == 2
        for weight in weights:
            assert weight.dtype == np.float32
            assert np.all(weight >= 0)
            assert weight.shape == (16, 200, 2)

    def test_edge_case_zero_weights(self, sample_weighted_iris_template_data):
        """Test that zero weights are allowed."""
        # Create weights with some zero values
        zero_weights = [
            np.zeros((16, 200, 2), dtype=np.float32),
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
        ]

        weighted_template = dc.WeightedIrisTemplate(
            iris_codes=sample_weighted_iris_template_data["iris_codes"],
            mask_codes=sample_weighted_iris_template_data["mask_codes"],
            weights=zero_weights,
            iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
        )

        assert np.all(weighted_template.weights[0] == 0)
        assert np.all(weighted_template.weights[1] >= 0)

    def test_edge_case_single_wavelet(self):
        """Test WeightedIrisTemplate with a single wavelet."""
        iris_codes = [np.random.choice(2, size=(16, 200, 2)).astype(bool)]
        mask_codes = [np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool)]
        iris_code_version = "v2.1"

        weighted_template = dc.WeightedIrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            weights=[np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32)],
            iris_code_version=iris_code_version,
        )

        assert len(weighted_template.iris_codes) == 1
        assert len(weighted_template.mask_codes) == 1
        assert len(weighted_template.weights) == 1
        assert weighted_template.iris_code_version == iris_code_version

    def test_from_iris_template(self, sample_weighted_iris_template_data):
        """Test the from_iris_template static method."""
        # Create an IrisTemplate first
        iris_template = dc.IrisTemplate(
            iris_codes=sample_weighted_iris_template_data["iris_codes"],
            mask_codes=sample_weighted_iris_template_data["mask_codes"],
            iris_code_version=sample_weighted_iris_template_data["iris_code_version"],
        )

        # Create weights
        weights = sample_weighted_iris_template_data["weights"]

        # Create WeightedIrisTemplate using from_iris_template
        weighted_template = dc.WeightedIrisTemplate.from_iris_template(iris_template, weights)

        # Check that the weighted template has the correct properties
        assert len(weighted_template.iris_codes) == len(iris_template.iris_codes)
        assert len(weighted_template.mask_codes) == len(iris_template.mask_codes)
        assert len(weighted_template.weights) == len(weights)
        assert weighted_template.iris_code_version == iris_template.iris_code_version

        # Check that the data is preserved
        for i in range(len(iris_template.iris_codes)):
            np.testing.assert_array_equal(weighted_template.iris_codes[i], iris_template.iris_codes[i])
            np.testing.assert_array_equal(weighted_template.mask_codes[i], iris_template.mask_codes[i])
            np.testing.assert_array_almost_equal(weighted_template.weights[i], weights[i])

        # Check that it's a proper WeightedIrisTemplate
        assert isinstance(weighted_template, dc.WeightedIrisTemplate)
        assert isinstance(weighted_template, dc.IrisTemplate)

    def test_from_iris_template_with_different_weights(self):
        """Test from_iris_template with different weight configurations."""
        # Create an IrisTemplate
        iris_codes = [
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
        ]
        mask_codes = [
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
        ]
        iris_code_version = "v2.1"

        iris_template = dc.IrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            iris_code_version=iris_code_version,
        )

        # Test with uniform weights
        uniform_weights = [
            np.ones((16, 200, 2), dtype=np.float32),
            np.ones((16, 200, 2), dtype=np.float32),
        ]

        weighted_template_uniform = dc.WeightedIrisTemplate.from_iris_template(iris_template, uniform_weights)
        assert np.all(weighted_template_uniform.weights[0] == 1.0)
        assert np.all(weighted_template_uniform.weights[1] == 1.0)

        # Test with zero weights
        zero_weights = [
            np.zeros((16, 200, 2), dtype=np.float32),
            np.zeros((16, 200, 2), dtype=np.float32),
        ]

        weighted_template_zero = dc.WeightedIrisTemplate.from_iris_template(iris_template, zero_weights)
        assert np.all(weighted_template_zero.weights[0] == 0.0)
        assert np.all(weighted_template_zero.weights[1] == 0.0)

        # Test with random weights
        random_weights = [
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
        ]

        weighted_template_random = dc.WeightedIrisTemplate.from_iris_template(iris_template, random_weights)
        assert np.all(weighted_template_random.weights[0] >= 0)
        assert np.all(weighted_template_random.weights[1] >= 0)

    def test_from_iris_template_validation(self):
        """Test that from_iris_template applies the same validation as direct initialization."""
        # Create an IrisTemplate
        iris_codes = [
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
        ]
        mask_codes = [
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
        ]
        iris_code_version = "v2.1"

        iris_template = dc.IrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            iris_code_version=iris_code_version,
        )

        # Test with wrong length weights
        wrong_length_weights = [np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32)]  # Only one weight

        with pytest.raises(ValueError, match="weights and iris_codes must have same length"):
            dc.WeightedIrisTemplate.from_iris_template(iris_template, wrong_length_weights)

        # Test with wrong shape weights
        wrong_shape_weights = [
            np.random.uniform(0.0, 1.0, size=(16, 100, 2)).astype(np.float32),  # Wrong width
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
        ]

        with pytest.raises(ValueError, match="Shape mismatch at wavelet 0"):
            dc.WeightedIrisTemplate.from_iris_template(iris_template, wrong_shape_weights)

        # Test with negative weights
        negative_weights = [
            np.random.uniform(-1.0, 1.0, size=(16, 200, 2)).astype(np.float32),  # Contains negative values
            np.random.uniform(0.0, 1.0, size=(16, 200, 2)).astype(np.float32),
        ]

        with pytest.raises(ValueError, match="All weights must be >= 0"):
            dc.WeightedIrisTemplate.from_iris_template(iris_template, negative_weights)

        # Test with non-float weights
        int_weights = [
            np.random.randint(0, 2, size=(16, 200, 2)).astype(np.int32),  # Integer instead of float
            np.random.randint(0, 2, size=(16, 200, 2)).astype(np.int32),
        ]

        with pytest.raises(ValidationError, match="Weight must be float array"):
            dc.WeightedIrisTemplate.from_iris_template(iris_template, int_weights)

    def test_as_iris_template(self, weighted_iris_template):
        """Test conversion to IrisTemplate using as_iris_template."""
        iris_template = weighted_iris_template.as_iris_template()
        assert isinstance(iris_template, dc.IrisTemplate)
        assert not hasattr(iris_template, "weights")
        assert iris_template.iris_code_version == weighted_iris_template.iris_code_version
        assert len(iris_template.iris_codes) == len(weighted_iris_template.iris_codes)
        assert len(iris_template.mask_codes) == len(weighted_iris_template.mask_codes)
        for arr1, arr2 in zip(iris_template.iris_codes, weighted_iris_template.iris_codes):
            np.testing.assert_array_equal(arr1, arr2)
        for arr1, arr2 in zip(iris_template.mask_codes, weighted_iris_template.mask_codes):
            np.testing.assert_array_equal(arr1, arr2)
