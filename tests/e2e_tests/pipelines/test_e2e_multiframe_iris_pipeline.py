import json
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
from iris.orchestration.output_builders import build_simple_multiframe_iris_pipeline_output
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
        aggregation_pipeline_output = aggregation_pipeline.run(
            [IRImage(img_data=ir_image, image_id="image_id", eye_side="right")]
        )
        # Assert the structure of the output based on the output_builders

        # The output should be a dict with keys: error, iris_template, metadata, individual_frames, templates_aggregation_metadata
        assert isinstance(aggregation_pipeline_output, dict)
        assert "error" in aggregation_pipeline_output
        assert "iris_template" in aggregation_pipeline_output
        assert "metadata" in aggregation_pipeline_output
        assert "individual_frames" in aggregation_pipeline_output
        assert "templates_aggregation_metadata" in aggregation_pipeline_output
        assert aggregation_pipeline_output["metadata"]["eye_side"] == "right"

        # check that metadata is json serializable
        json.dumps(aggregation_pipeline_output["metadata"])
        for frame in aggregation_pipeline_output["individual_frames"]:
            json.dumps(frame["metadata"])
        json.dumps(aggregation_pipeline_output["templates_aggregation_metadata"])

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
        assert "image_id" in aggregation_pipeline_output["individual_frames"][0]["metadata"].keys()

        # check that the individiual frame metadata contains all the fields from IrisPipeline metadata
        dummy_call_trace = Mock()
        # Create a mock IRImage-like object with the required properties
        mock_ir_image = Mock()
        mock_ir_image.width = ir_image.shape[1]
        mock_ir_image.height = ir_image.shape[0]
        mock_ir_image.eye_side = "right"
        mock_ir_image.image_id = "image_id"
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

    @pytest.mark.parametrize(
        "image_ids,expected_behavior",
        [
            # Test case 1: Valid strings
            (["image_001", "img_alpha_beta", "test-image_2024", "normal123"], "valid_strings"),
            # Test case 2: Invalid/Edge case strings
            (["", "a" * 1000, "test\n\r\t", "🔍🎯"], "invalid_strings"),
            # Test case 3: All Nones
            ([None, None, None, None], "all_nones"),
            # Test case 4: Combination of strings and Nones
            (["image_1", None, "image_3", None], "mixed_strings_nones"),
            # Test case 5: Duplicated strings
            (["duplicate", "duplicate", "duplicate", "unique"], "duplicated_strings"),
            # Test case 6: Conflict with auto-generated frame IDs
            (["frame_1", None, "frame_3", None, None], "frame_conflicts"),
            # Test case 7: Multiple levels of frame ID conflicts
            (["frame_1", None, "frame_1_1", None, None], "deep_frame_conflicts"),
        ],
        ids=[
            "valid_strings",
            "invalid_strings",
            "all_nones",
            "mixed_strings_nones",
            "duplicated_strings",
            "frame_conflicts",
            "deep_frame_conflicts",
        ],
    )
    def test_iris_pipeline_image_id_assignments(self, image_ids, expected_behavior, ir_image):
        """Test the iris pipeline with different image_id assignment scenarios."""
        combined_config = load_yaml_config(MultiframeIrisPipeline.DEFAULT_PIPELINE_CFG_PATH)
        env = MultiframeIrisPipeline.ORB_ENVIRONMENT

        aggregation_pipeline = MultiframeIrisPipeline(config=combined_config, env=env)

        # Create IRImages with specified image_ids
        images = []
        for i, image_id in enumerate(image_ids):
            # Use the same image data for all frames
            images.append(IRImage(img_data=ir_image, image_id=image_id, eye_side="right"))

        # Run the pipeline
        aggregation_pipeline_output = aggregation_pipeline.run(images)

        # Basic assertions for all cases
        assert isinstance(aggregation_pipeline_output, dict)
        assert "error" in aggregation_pipeline_output
        assert "individual_frames" in aggregation_pipeline_output
        assert len(aggregation_pipeline_output["individual_frames"]) == len(image_ids)

        # Extract the resulting image_ids from individual frames metadata
        individual_frames_image_ids = []
        for frame in aggregation_pipeline_output["individual_frames"]:
            assert "metadata" in frame
            assert "image_id" in frame["metadata"]
            individual_frames_image_ids.append(frame["metadata"]["image_id"])

        # Individual frames image_ids should be the same as the input image_ids
        for original, result in zip(image_ids, individual_frames_image_ids):
            assert result == original, f"Expected {original}, got {result}"

        # extract image_ids from aggregation pipeline output
        aggregation_pipeline_image_ids = aggregation_pipeline_output["templates_aggregation_metadata"]["metadata"][
            "input_templates_image_ids"
        ]

        if expected_behavior in ["valid_strings", "invalid_strings", "duplicated_strings"]:
            # Individual frames image_ids should be the same as the input image_ids
            for original, result in zip(image_ids, aggregation_pipeline_image_ids):
                assert result == original, f"Expected {original}, got {result}"

        elif expected_behavior == "all_nones":
            for i in range(len(image_ids)):
                assert (
                    aggregation_pipeline_image_ids[i] == f"frame_{i}"
                ), f"Expected 'frame_{i}', got {aggregation_pipeline_image_ids[i]}"

        elif expected_behavior == "mixed_strings_nones":
            # Strings should be preserved, Nones should remain as None
            for i, (original, result) in enumerate(zip(image_ids, aggregation_pipeline_image_ids)):
                if original is not None:
                    assert result == original, f"Expected {original}, got {result}"
                else:
                    assert result == f"frame_{i}", f"Expected 'frame_{i}', got {result}"

        elif expected_behavior == "frame_conflicts":
            # Test collision handling: existing strings preserved, None values get unique IDs
            # Expected: ["frame_1", None, "frame_3", None, None]
            # Should become: ["frame_1", "frame_1_1", "frame_3", "frame_3_1", "frame_4"]
            expected_results = ["frame_1", "frame_1_1", "frame_3", "frame_3_1", "frame_4"]

            for expected, actual in zip(expected_results, aggregation_pipeline_image_ids):
                assert actual == expected, f"Expected {expected}, got {actual}"

            # Verify all IDs are unique
            assert len(set(aggregation_pipeline_image_ids)) == len(
                aggregation_pipeline_image_ids
            ), "All aggregation IDs should be unique"

        elif expected_behavior == "deep_frame_conflicts":
            # Test deep collision handling: multiple levels of conflicts
            # Input: ["frame_1", "frame_1_1", None, None, None]
            # Should become: ["frame_1", "frame_1_1", "frame_2", "frame_3", "frame_4"]
            # Note: at i=1 frame_1 is already taken, so None tries to get frame_1_1, which is already taken, so it tries to get frame_1_2, etc.
            expected_results = ["frame_1", "frame_1_2", "frame_1_1", "frame_3", "frame_4"]

            for expected, actual in zip(expected_results, aggregation_pipeline_image_ids):
                assert actual == expected, f"Expected {expected}, got {actual}"

            # Verify all IDs are unique
            assert len(set(aggregation_pipeline_image_ids)) == len(
                aggregation_pipeline_image_ids
            ), "All aggregation IDs should be unique"
