from unittest import mock

import numpy as np
import pytest

from iris.io.dataclasses import IRImage, SegmentationMap
from iris.nodes.segmentation.multilabel_segmentation_interface import MultilabelSemanticSegmentationInterface
from iris.nodes.segmentation.onnx_multilabel_segmentation import ONNXMultilabelSegmentation


@pytest.fixture
def multilabel_model() -> ONNXMultilabelSegmentation:
    return ONNXMultilabelSegmentation()


def test_forward(multilabel_model: ONNXMultilabelSegmentation) -> None:
    mock_grayscale_image_data = np.ones((1440, 1080), dtype=np.uint) * 255
    mock_irimage = IRImage(img_data=mock_grayscale_image_data, eye_side="left")

    preprocessed_input = multilabel_model._preprocess(image=mock_irimage.img_data)
    predictions = multilabel_model._forward(preprocessed_input)

    assert len(predictions) == 1

    geometry_pred = predictions[0]
    assert geometry_pred.shape == (
        1,
        len(MultilabelSemanticSegmentationInterface.CLASSES_MAPPING),
        480,
        640,
    )


def test_postprocess_segmap(multilabel_model: ONNXMultilabelSegmentation) -> None:
    mock_predicted_segmap = np.ones((1, 2, 512, 512)) * 0.5
    mock_original_input_image_resolution = (1080, 1440)

    expected_output = np.ones((1440, 1080, 2)) * 0.5

    postprocess_segmap = multilabel_model.postprocess_segmap(
        mock_predicted_segmap, mock_original_input_image_resolution
    )

    np.testing.assert_allclose(postprocess_segmap, expected_output)


def test_postprocess(multilabel_model: ONNXMultilabelSegmentation) -> None:
    mock_prediction = [np.ones((1, 4, 512, 512)) * 0.25, np.ones((1, 2, 512, 512)) * 0.5]
    mock_original_input_image_resolution = (1080, 1440)

    expected_output = SegmentationMap(
        predictions=np.ones((1440, 1080, 4)) * 0.25,
        index2class=MultilabelSemanticSegmentationInterface.CLASSES_MAPPING,
    )

    postprocess_output = multilabel_model._postprocess(mock_prediction, mock_original_input_image_resolution)

    assert postprocess_output == expected_output


def test_run(multilabel_model: ONNXMultilabelSegmentation) -> None:
    mock_grayscale_image_data = np.ones((1440, 1080), dtype=np.uint) * 255
    mock_irimage = IRImage(img_data=mock_grayscale_image_data, eye_side="left")

    mock_prediction = [
        np.ones((1, 4, 512, 512)) * 0.5,
        np.ones((1, 4, 512, 512)) * 0.5,
        np.ones((1, 4, 512, 512)) * 0.5,
        np.ones((1, 2, 512, 512)) * 0.5,
    ]

    expected_output = SegmentationMap(
        predictions=np.ones((1440, 1080, 4)) * 0.5,
        index2class=MultilabelSemanticSegmentationInterface.CLASSES_MAPPING,
    )

    with mock.patch(
        "iris.nodes.segmentation.onnx_multilabel_segmentation.ONNXMultilabelSegmentation._forward"
    ) as model_mock:
        model_mock.return_value = mock_prediction

        output = multilabel_model(mock_irimage)

        assert output == expected_output
