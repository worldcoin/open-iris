import os
import random
from typing import List
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import IRImage, IrisTemplate
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import __get_iris_pipeline_metadata as get_iris_pipeline_metadata
from iris.orchestration.output_builders import (
    __get_templates_aggregation_metadata as get_templates_aggregation_metadata,
)
from iris.orchestration.output_builders import (
    build_simple_multiframe_iris_pipeline_output,
)
from iris.pipelines.base_pipeline import load_yaml_config
from iris.pipelines.multiframe_iris_pipeline import MultiframeIrisPipeline


def assert_pipeline_output(output: dict, env: Environment):
    """Assert the structure of the output based on the output_builders."""
    assert isinstance(output, dict)
    assert "error" in output
    assert "iris_template" in output
    assert "metadata" in output
    assert "individual_frames" in output
    assert "templates_aggregation_metadata" in output


@pytest.fixture
def ir_image() -> np.ndarray:
    ir_image_path = os.path.join(os.path.dirname(__file__), "mocks", "inputs", "anonymized.png")
    img_data = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
    return img_data


@pytest.fixture
def ir_images(request, ir_image) -> List[np.ndarray]:
    # Get number of images from marker, default to 3
    nb_images = getattr(request.node.get_closest_marker("nb_images"), "args", [3])[0]

    images = []
    h, w = ir_image.shape

    for _ in range(nb_images):
        # --- Random rotation by -7 to +7 degrees ---
        angle = random.uniform(-7, 7)
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(ir_image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        # --- Randomly modify a few random bits/pixels ---
        noisy = rotated.copy()
        n_noisy_pixels = max(1, (w * h) // 200)  # e.g. small percentage of the image
        ys = np.random.randint(0, h, n_noisy_pixels)
        xs = np.random.randint(0, w, n_noisy_pixels)
        for x, y in zip(xs, ys):
            noisy[y, x] = np.clip(noisy[y, x] + np.random.randint(-8, 9), 0, 255)  # small change

        images.append(noisy)

    return images


class TestMultiframeIrisPipeline:
    @pytest.mark.parametrize(
        "env",
        [
            Environment(
                pipeline_output_builder=build_simple_multiframe_iris_pipeline_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            MultiframeIrisPipeline.ORB_ENVIRONMENT,
        ],
        ids=["simple_env", "orb_env"],
    )
    def test_iris_pipeline_with_aggregation_single_image(self, env, ir_image):
        """Test the iris pipeline with aggregation using different environment combinations."""
        combined_config = load_yaml_config(MultiframeIrisPipeline.DEFAULT_PIPELINE_CFG_PATH)

        aggregation_pipeline = MultiframeIrisPipeline(config=combined_config, env=env)
        aggregation_pipeline_output = aggregation_pipeline.run([IRImage(img_data=ir_image, image_id="image_id", eye_side="right")])
        # Assert the structure of the output based on the output_builders

        # The output should be a dict with keys: error, iris_template, metadata, individual_frames, templates_aggregation_metadata
        assert isinstance(aggregation_pipeline_output, dict)
        assert "error" in aggregation_pipeline_output
        assert "iris_template" in aggregation_pipeline_output
        assert "metadata" in aggregation_pipeline_output
        assert "individual_frames" in aggregation_pipeline_output
        assert "templates_aggregation_metadata" in aggregation_pipeline_output

        # Check types of the main fields
        # error can be None or dict
        assert aggregation_pipeline_output["error"] is None

        # iris_template can a dict or IrisTemplate (depending on serialization)
        if env == MultiframeIrisPipeline.ORB_ENVIRONMENT:
            assert isinstance(aggregation_pipeline_output["iris_template"], (dict))
        else:
            assert isinstance(aggregation_pipeline_output["iris_template"], IrisTemplate)
        # metadata should be a dict
        assert isinstance(aggregation_pipeline_output["metadata"], dict)

        # individual_frames should be a list (of dicts)
        assert isinstance(aggregation_pipeline_output["individual_frames"], list)
        assert len(aggregation_pipeline_output["individual_frames"]) == 1
        assert isinstance(aggregation_pipeline_output["individual_frames"][0], dict)
        assert "error" in aggregation_pipeline_output["individual_frames"][0]
        assert "metadata" in aggregation_pipeline_output["individual_frames"][0]
        assert isinstance(aggregation_pipeline_output["individual_frames"][0]["metadata"]["image_size"], tuple)

        # check that the individiual frame metadata contains all the fields from IrisPipeline metadata
        dummy_call_trace = Mock()
        # Create a mock IRImage-like object with the required properties
        mock_ir_image = Mock()
        mock_ir_image.width = ir_image.shape[1]
        mock_ir_image.height = ir_image.shape[0]
        mock_ir_image.eye_side = "right"
        dummy_call_trace.get_input.return_value = mock_ir_image
        dummy_call_trace.get.side_effect = lambda k: None
        expected_keys = set(get_iris_pipeline_metadata(dummy_call_trace).keys())
        actual_keys = set(aggregation_pipeline_output["individual_frames"][0]["metadata"].keys())
        assert expected_keys == actual_keys

        # templates_aggregation_metadata should be a dict
        assert isinstance(aggregation_pipeline_output["templates_aggregation_metadata"], dict)

        # Check some expected metadata fields
        metadata = aggregation_pipeline_output["metadata"]
        assert "iris_version" in metadata
        assert "input_images_count" in metadata
        assert "eye_side" in metadata
        assert "aggregation_successful" in metadata
        assert "is_aggregated" in metadata

        # check that templates_aggregation_metadata contains all the metadata from TemplatesAggregationPipeline
        dummy_call_trace = Mock()
        dummy_call_trace.get_input.return_value = [None]
        dummy_call_trace.get.side_effect = lambda k: None
        expected_keys = set(get_templates_aggregation_metadata(dummy_call_trace).keys())
        actual_keys = set(aggregation_pipeline_output["templates_aggregation_metadata"]["metadata"].keys())
        assert expected_keys == actual_keys

    @pytest.mark.parametrize(
        "env",
        [
            Environment(
                pipeline_output_builder=build_simple_multiframe_iris_pipeline_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            MultiframeIrisPipeline.ORB_ENVIRONMENT,
        ],
        ids=["simple_env", "orb_env"],
    )
    @pytest.mark.nb_images(3)
    def test_iris_pipeline_with_aggregation_multiple_images(self, env, ir_images):
        """Test the iris pipeline with aggregation using different environment combinations."""
        combined_config = load_yaml_config(MultiframeIrisPipeline.DEFAULT_PIPELINE_CFG_PATH)

        aggregation_pipeline = MultiframeIrisPipeline(config=combined_config, env=env)
        images = [IRImage(img_data=img, image_id=f"image_{i}", eye_side="right") for i, img in enumerate(ir_images)]
        aggregation_pipeline_output = aggregation_pipeline.run(images)

        # Assert the structure of the output based on the output_builders
        assert isinstance(aggregation_pipeline_output, dict)
        assert aggregation_pipeline_output["error"] is None
        if env == MultiframeIrisPipeline.ORB_ENVIRONMENT:
            assert isinstance(aggregation_pipeline_output["iris_template"], (dict))
        else:
            assert isinstance(aggregation_pipeline_output["iris_template"], IrisTemplate)

        assert isinstance(aggregation_pipeline_output["metadata"], dict)
        assert isinstance(aggregation_pipeline_output["individual_frames"], list)
        assert len(aggregation_pipeline_output["individual_frames"]) == len(ir_images)
        assert isinstance(aggregation_pipeline_output["templates_aggregation_metadata"], dict)

        assert aggregation_pipeline_output["templates_aggregation_metadata"]["metadata"][
            "post_identity_filter_templates_count"
        ] == len(ir_images)
