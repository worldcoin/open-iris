import os

import cv2
import numpy as np
import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import IRImage, IrisTemplate
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import raise_error_manager, store_error_manager
from iris.orchestration.output_builders import (
    build_simple_iris_pipeline_debugging_output,
    build_simple_templates_aggregation_output,
)
from iris.pipelines.base_pipeline import load_yaml_config
from iris.pipelines.iris_pipeline import IRISPipeline
from iris.pipelines.templates_aggregation_pipeline import TemplatesAggregationPipeline


@pytest.fixture
def random_iris_templates():
    """Create a set of sample IrisTemplates for end-to-end testing."""
    templates = []

    # Create 5 templates with consistent structure but different content
    for i in range(5):
        # Create iris codes with some variation
        iris_codes = []
        mask_codes = []

        for wavelet in range(2):  # 2 wavelets
            # Create patterns that will have some consistency for majority voting
            if i < 3:  # First 3 templates have similar patterns
                iris_code = np.random.choice(2, size=(8, 32, 2), p=[0.3, 0.7]).astype(bool)
            else:  # Last 2 templates have different patterns
                iris_code = np.random.choice(2, size=(8, 32, 2), p=[0.7, 0.3]).astype(bool)

            # Create masks with high validity
            mask_code = np.random.choice(2, size=(8, 32, 2), p=[0.1, 0.9]).astype(bool)

            iris_codes.append(iris_code)
            mask_codes.append(mask_code)

        template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
        templates.append(template)

    return templates


@pytest.fixture
def incompatible_iris_templates():
    """Create a set of incompatible IrisTemplates for testing error cases."""
    # Template 1: Standard template
    iris_codes_1 = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
    mask_codes_1 = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
    template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

    # Template 2: Different iris code version
    iris_codes_2 = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
    mask_codes_2 = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
    template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v3.0")

    return [template1, template2]


@pytest.fixture
def standalone_aggregation_config():
    """Create a custom configuration for the aggregation pipeline (updated to match YAML)."""
    return {
        "metadata": {"pipeline_name": "templates_aggregation", "iris_version": "1.7.1"},
        "pipeline": [
            {
                "name": "templates_alignment",
                "algorithm": {
                    "class_name": "iris.nodes.templates_alignment.hamming_distance_based.HammingDistanceBasedAlignment",
                    "params": {
                        "rotation_shift": 15,
                        "use_first_as_reference": False,
                        "normalise": True,
                        "reference_selection_method": "linear",
                    },
                },
                "inputs": [{"name": "templates_with_ids", "source_node": "input"}],
                "callbacks": [],
            },
            {
                "name": "identity_validation",
                "algorithm": {
                    "class_name": "iris.nodes.templates_filter.single_identity_filter.TemplateIdentityFilter",
                    "params": {
                        "identity_distance_threshold": 0.35,
                        "identity_validation_action": "remove",
                        "min_templates_after_validation": 1,
                    },
                },
                "inputs": [{"name": "aligned_templates", "source_node": "templates_alignment"}],
                "callbacks": [],
            },
            {
                "name": "templates_aggregation",
                "algorithm": {
                    "class_name": "iris.nodes.templates_aggregation.majority_vote.MajorityVoteAggregation",
                    "params": {
                        "consistency_threshold": 0.75,
                        "mask_threshold": 0.01,
                        "use_inconsistent_bits": True,
                        "inconsistent_bit_threshold": 0.4,
                    },
                },
                "inputs": [{"name": "templates", "source_node": "identity_validation"}],
                "callbacks": [
                    {
                        "class_name": "iris.nodes.validators.object_validators.AreTemplatesAggregationCompatible",
                        "params": {},
                    }
                ],
            },
        ],
    }


@pytest.fixture
def composite_iris_config():
    """Create a custom configuration for the aggregation pipeline (updated to match YAML)."""
    return {
        "metadata": {"pipeline_name": "iris_pipeline", "iris_version": "1.7.2"},
        "pipeline": [],
        "template_aggregation": {
            "metadata": {"pipeline_name": "templates_aggregation", "iris_version": "1.7.1"},
            "pipeline": [
                {
                    "name": "templates_alignment",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_alignment.hamming_distance_based.HammingDistanceBasedAlignment",
                        "params": {
                            "rotation_shift": 15,
                            "use_first_as_reference": False,
                            "normalise": True,
                            "reference_selection_method": "linear",
                        },
                    },
                    "inputs": [{"name": "templates_with_ids", "source_node": "input"}],
                    "callbacks": [],
                },
                {
                    "name": "identity_validation",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_filter.single_identity_filter.TemplateIdentityFilter",
                        "params": {
                            "identity_distance_threshold": 0.35,
                            "identity_validation_action": "remove",
                            "min_templates_after_validation": 1,
                        },
                    },
                    "inputs": [{"name": "aligned_templates", "source_node": "templates_alignment"}],
                    "callbacks": [],
                },
                {
                    "name": "templates_aggregation",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_aggregation.majority_vote.MajorityVoteAggregation",
                        "params": {
                            "consistency_threshold": 0.75,
                            "mask_threshold": 0.01,
                            "use_inconsistent_bits": True,
                            "inconsistent_bit_threshold": 0.4,
                        },
                    },
                    "inputs": [{"name": "templates", "source_node": "identity_validation"}],
                    "callbacks": [
                        {
                            "class_name": "iris.nodes.validators.object_validators.AreTemplatesAggregationCompatible",
                            "params": {},
                        }
                    ],
                },
            ],
        },
    }


@pytest.fixture
def ir_image() -> np.ndarray:
    ir_image_path = os.path.join(os.path.dirname(__file__), "mocks", "inputs", "anonymized.png")
    img_data = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
    return img_data


@pytest.fixture
def same_id_iris_templates():
    """Create a set of IrisTemplates with very small Hamming distances between them (including mask codes)."""
    num_templates = 5
    num_wavelets = 2
    shape = (8, 32, 2)
    iris_code_version = "v2.1"
    templates = []

    # Create a base template
    base_iris_codes = [np.random.choice(2, size=shape).astype(bool) for _ in range(num_wavelets)]
    base_mask_codes = [np.random.choice(2, size=shape, p=[0.1, 0.9]).astype(bool) for _ in range(num_wavelets)]

    for i in range(num_templates):
        # Copy base codes
        iris_codes = [arr.copy() for arr in base_iris_codes]
        mask_codes = [arr.copy() for arr in base_mask_codes]

        # Flip a small number of random bits in each wavelet's iris code and mask code
        for w in range(num_wavelets):
            n_flips_iris = 2  # Number of bits to flip per wavelet in iris code
            n_flips_mask = 1  # Number of bits to flip per wavelet in mask code

            # Flip iris code bits
            flat_iris = iris_codes[w].flatten()
            flip_indices_iris = np.random.choice(flat_iris.size, size=n_flips_iris, replace=False)
            flat_iris[flip_indices_iris] = ~flat_iris[flip_indices_iris]
            iris_codes[w] = flat_iris.reshape(shape)

            # Flip mask code bits
            flat_mask = mask_codes[w].flatten()
            flip_indices_mask = np.random.choice(flat_mask.size, size=n_flips_mask, replace=False)
            flat_mask[flip_indices_mask] = ~flat_mask[flip_indices_mask]
            mask_codes[w] = flat_mask.reshape(shape)

        template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version=iris_code_version)
        templates.append(template)

    return templates


class TestIRISPipelineWithAggregation:
    """End-to-end tests for iris pipeline with aggregation functionality."""

    @pytest.mark.parametrize(
        "iris_env,aggregation_env",
        [
            # Standard predefined environments
            (IRISPipeline.ORB_ENVIRONMENT, TemplatesAggregationPipeline.ORB_ENVIRONMENT),
            #
            (
                Environment(
                    pipeline_output_builder=build_simple_iris_pipeline_debugging_output,
                    error_manager=raise_error_manager,
                    call_trace_initialiser=PipelineCallTraceStorage.initialise,
                ),
                Environment(
                    pipeline_output_builder=build_simple_templates_aggregation_output,
                    error_manager=raise_error_manager,
                    call_trace_initialiser=PipelineCallTraceStorage.initialise,
                ),
            ),
        ],
        ids=[
            "iris_orb_aggregation_orb",
            "iris_simple_aggregation_simple",
        ],
    )
    def test_iris_pipeline_with_aggregation(self, ir_image, iris_env, aggregation_env):
        """Test the iris pipeline with aggregation using different environment combinations."""
        combined_config = load_yaml_config(TemplatesAggregationPipeline.DEFAULT_PIPELINE_CFG_PATH)

        aggregation_pipeline = TemplatesAggregationPipeline(
            config=combined_config, subconfig_key="templates_aggregation", env=aggregation_env
        )

        iris_pipeline = IRISPipeline(config=combined_config, env=iris_env)
        iris_templates = []
        for _ in range(3):
            iris_pipeline_output = iris_pipeline(IRImage(img_data=ir_image, image_id="image_id", eye_side="right"))
            if iris_env == IRISPipeline.ORB_ENVIRONMENT:
                template = IrisTemplate.deserialize(iris_pipeline_output["iris_template"])
            else:
                template = iris_pipeline_output["iris_template"]
            iris_templates.append(template)

        aggregation_pipeline_output = aggregation_pipeline.run(iris_templates)
        if aggregation_env == TemplatesAggregationPipeline.ORB_ENVIRONMENT:
            aggregated_template = IrisTemplate.deserialize(aggregation_pipeline_output["iris_template"])
        else:
            aggregated_template = aggregation_pipeline_output["iris_template"]
        assert aggregation_pipeline_output["error"] is None
        assert aggregation_pipeline_output["iris_template"] is not None
        assert aggregation_pipeline_output["metadata"] is not None

        assert all(
            np.array_equal(
                iris_templates[0].iris_codes[i] * iris_templates[0].mask_codes[i], aggregated_template.iris_codes[i]
            )
            for i in range(len(iris_templates[0].iris_codes))
        )


class TestTemplatesAggregationPipeline:
    """End-to-end tests for templates aggregation functionality."""

    @pytest.mark.parametrize(
        "config,subconfig_key",
        [
            ("standalone_aggregation_config", ""),
            ("composite_iris_config", "template_aggregation"),
        ],
        ids=["standalone aggregation", "composite iris pipeline"],
    )
    def test_full_pipeline_with_compatible_templates(self, config, subconfig_key, request, same_id_iris_templates):
        """Test the complete pipeline flow with compatible templates."""
        # Initialize the pipeline
        config = request.getfixturevalue(config)
        pipeline = TemplatesAggregationPipeline(config=config, subconfig_key=subconfig_key)

        # Run the pipeline
        result = pipeline.run(same_id_iris_templates)

        # Verify the output structure
        assert isinstance(result, dict)
        assert "iris_template" in result
        assert "error" in result
        assert "metadata" in result

        # Verify successful execution (no error)
        assert result["error"] is None

        # Verify the aggregated template
        aggregated_template = result["iris_template"]
        assert isinstance(aggregated_template, IrisTemplate)
        assert aggregated_template.iris_code_version == same_id_iris_templates[0].iris_code_version
        assert len(aggregated_template.iris_codes) == len(same_id_iris_templates[0].iris_codes)
        assert len(aggregated_template.mask_codes) == len(same_id_iris_templates[0].mask_codes)

        # Verify metadata
        metadata = result["metadata"]
        assert isinstance(metadata, dict)
        assert "input_templates_count" in metadata
        assert metadata["input_templates_count"] == len(same_id_iris_templates)
        assert "iris_version" in metadata
        assert "aligned_templates" in metadata
        assert "reference_template_id" in metadata["aligned_templates"]
        assert "distances" in metadata["aligned_templates"]
        assert "post_identity_filter_templates_count" in metadata
        assert metadata["post_identity_filter_templates_count"] == len(same_id_iris_templates)

    def test_pipeline_with_incompatible_templates(self, incompatible_iris_templates, standalone_aggregation_config):
        """Test the pipeline with incompatible templates (should fail validation)."""
        # Initialize the pipeline
        pipeline = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")

        # Run the pipeline - should handle the validation error
        result = pipeline.run(incompatible_iris_templates)

        # Verify that an error was captured
        assert result["error"] is not None
        assert result["iris_template"] is None

    def test_pipeline_with_single_template(self, standalone_aggregation_config):
        """Test the pipeline with a single template."""
        # Create a single template
        iris_codes = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
        mask_codes = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
        single_template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")

        # Initialize the pipeline
        pipeline = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")

        # Run the pipeline
        result = pipeline.run([single_template])

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None

        # For single template, the output should be essentially the same as input
        aggregated_template = result["iris_template"]
        assert aggregated_template.iris_code_version == single_template.iris_code_version
        assert len(aggregated_template.iris_codes) == len(single_template.iris_codes)

    def test_pipeline_with_default_configuration_same_id(self, same_id_iris_templates):
        """Test the pipeline using default configuration."""
        # Initialize with default config
        pipeline = TemplatesAggregationPipeline()

        # Run the pipeline
        result = pipeline.run(same_id_iris_templates)

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None

    def test_pipeline_with_custom_environment(self, same_id_iris_templates, standalone_aggregation_config):
        """Test the pipeline with a custom environment."""
        # Create custom environment
        custom_env = Environment(
            pipeline_output_builder=build_simple_templates_aggregation_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        )

        # Initialize pipeline with custom environment
        pipeline = TemplatesAggregationPipeline(config=standalone_aggregation_config, env=custom_env, subconfig_key="")

        # Run the pipeline
        result = pipeline.run(same_id_iris_templates)

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None

    def test_pipeline_with_orb_environment(self, same_id_iris_templates):
        """Test the pipeline with ORB environment."""
        # Use the predefined ORB environment
        pipeline = TemplatesAggregationPipeline(env=TemplatesAggregationPipeline.ORB_ENVIRONMENT)

        # Run the pipeline
        result = pipeline.run(same_id_iris_templates)

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None

    def test_pipeline_call_trace_functionality(self, same_id_iris_templates, standalone_aggregation_config):
        """Test that the pipeline call trace works correctly."""
        pipeline = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")

        # Run the pipeline
        _ = pipeline.run(same_id_iris_templates)

        # Verify call trace has been populated
        assert pipeline.call_trace is not None

        # Check that input was written to call trace
        input_data = pipeline.call_trace.get_input()
        # input_data are IrisTemplateWithId objects, same_id_iris_templates are IrisTemplate objects
        # Since IrisTemplateWithId now inherits from IrisTemplate, we can compare the iris_codes
        assert len(input_data) == len(same_id_iris_templates)
        for i, (input_template, expected_template) in enumerate(zip(input_data, same_id_iris_templates)):
            assert input_template.iris_code_version == expected_template.iris_code_version
            assert len(input_template.iris_codes) == len(expected_template.iris_codes)

        # Check that the aggregation node result is available
        aggregation_result = pipeline.call_trace["templates_aggregation"]
        assert aggregation_result is not None

    def test_pipeline_with_empty_templates_list(self, standalone_aggregation_config):
        """Test the pipeline with an empty templates list."""
        pipeline = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")

        # Run with empty list - should be handled by validation
        result = pipeline.run([])

        # Should have an error due to validation failure
        assert result["error"] is not None
        assert result["iris_template"] is None

    def test_integration_with_many_templates(self, standalone_aggregation_config):
        """Test integration with a large number of templates."""
        # Create many compatible templates
        templates = []
        for i in range(20):  # Large number of templates
            iris_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool) for _ in range(1)]
            mask_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool) for _ in range(1)]
            template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
            templates.append(template)

        pipeline = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")
        result = pipeline.run(templates)

        # Should handle many templates
        assert result["error"] is None
        assert result["iris_template"] is not None
        assert result["metadata"]["input_templates_count"] == 20

    def test_error_handling_and_recovery(self, standalone_aggregation_config):
        """Test error handling in the pipeline."""
        pipeline = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")

        # Test with None input (should be handled gracefully)
        try:
            result = pipeline.run(None)
            # Should either work or produce a meaningful error
            assert "error" in result
        except Exception as e:
            # If it raises an exception, it should be a meaningful one
            assert isinstance(e, (TypeError, AttributeError, ValueError))

    def test_pipeline_reproducibility(self, standalone_aggregation_config):
        """Test that the pipeline produces consistent results."""
        # Create deterministic templates
        np.random.seed(42)
        templates = []
        for i in range(3):
            iris_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool)]
            mask_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool)]
            template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
            templates.append(template)

        # Run pipeline twice
        pipeline1 = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")
        result1 = pipeline1.run(templates)

        pipeline2 = TemplatesAggregationPipeline(config=standalone_aggregation_config, subconfig_key="")
        result2 = pipeline2.run(templates)

        # Results should be identical
        assert result1["error"] == result2["error"]
        if result1["iris_template"] is not None and result2["iris_template"] is not None:
            # Compare iris codes
            for i in range(len(result1["iris_template"].iris_codes)):
                np.testing.assert_array_equal(
                    result1["iris_template"].iris_codes[i], result2["iris_template"].iris_codes[i]
                )
                np.testing.assert_array_equal(
                    result1["iris_template"].mask_codes[i], result2["iris_template"].mask_codes[i]
                )


class TestE2ETemplatesAggregationFailures:
    """End-to-end tests for templates aggregation pipeline failure scenarios."""

    @pytest.fixture
    def failure_prone_config_no_clusters(self):
        """Configuration that will cause no identity clusters to be found."""
        return {
            "metadata": {"pipeline_name": "templates_aggregation", "iris_version": "1.7.1"},
            "pipeline": [
                {
                    "name": "templates_alignment",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_alignment.hamming_distance_based.HammingDistanceBasedAlignment",
                        "params": {
                            "rotation_shift": 15,
                            "use_first_as_reference": False,
                            "normalise": True,
                            "reference_selection_method": "linear",
                        },
                    },
                    "inputs": [{"name": "templates_with_ids", "source_node": "input"}],
                    "callbacks": [],
                },
                {
                    "name": "identity_validation",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_filter.single_identity_filter.TemplateIdentityFilter",
                        "params": {
                            "identity_distance_threshold": 0.05,  # Very low threshold
                            "identity_validation_action": "raise_error",  # Will raise error
                            "min_templates_after_validation": 1,
                        },
                    },
                    "inputs": [{"name": "aligned_templates", "source_node": "templates_alignment"}],
                    "callbacks": [],
                },
                {
                    "name": "templates_aggregation",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_aggregation.majority_vote.MajorityVoteAggregation",
                        "params": {
                            "consistency_threshold": 0.75,
                            "mask_threshold": 0.01,
                            "use_inconsistent_bits": True,
                            "inconsistent_bit_threshold": 0.4,
                        },
                    },
                    "inputs": [{"name": "templates", "source_node": "identity_validation"}],
                    "callbacks": [],
                },
            ],
        }

    @pytest.fixture
    def very_different_templates(self):
        """Create templates that are very different from each other (high Hamming distances)."""
        templates = []

        for i in range(4):
            iris_codes = []
            mask_codes = []

            for wavelet in range(2):
                # Create very different patterns for each template
                if i == 0:
                    # Template 0: mostly zeros
                    iris_code = np.zeros((8, 32, 2), dtype=bool)
                elif i == 1:
                    # Template 1: mostly ones
                    iris_code = np.ones((8, 32, 2), dtype=bool)
                elif i == 2:
                    # Template 2: checkerboard pattern
                    iris_code = np.zeros((8, 32, 2), dtype=bool)
                    iris_code[::2, ::2] = True
                    iris_code[1::2, 1::2] = True
                else:
                    # Template 3: inverse checkerboard
                    iris_code = np.ones((8, 32, 2), dtype=bool)
                    iris_code[::2, ::2] = False
                    iris_code[1::2, 1::2] = False

                # All templates have good masks
                mask_code = np.ones((8, 32, 2), dtype=bool)

                iris_codes.append(iris_code)
                mask_codes.append(mask_code)

            template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
            templates.append(template)

        return templates

    def test_e2e_no_identity_clusters_found(self, failure_prone_config_no_clusters, very_different_templates):
        """Test end-to-end pipeline when no identity clusters are found."""
        env = Environment(
            pipeline_output_builder=build_simple_templates_aggregation_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        )

        pipeline = TemplatesAggregationPipeline(config=failure_prone_config_no_clusters, env=env, subconfig_key="")

        image_ids = [f"image_{i}" for i in range(len(very_different_templates))]
        result = pipeline.run(very_different_templates, image_ids)

        # Should have error due to no clusters found
        assert result["error"] is not None
        assert "no identity clusters" in result["error"]["message"]
        assert result["iris_template"] is None

        # Metadata should still be generated
        assert result["metadata"] is not None
        assert result["metadata"]["input_templates_count"] == len(very_different_templates)

    def test_e2e_empty_templates_list(self):
        """Test end-to-end pipeline with empty templates list."""
        env = Environment(
            pipeline_output_builder=build_simple_templates_aggregation_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        )

        pipeline = TemplatesAggregationPipeline(env=env, subconfig_key="")

        result = pipeline.run(templates=[], image_ids=["image1"])

        assert result["error"] is not None
        assert "must match number of templates" in result["error"]["message"]
        assert result["iris_template"] is None
        assert result["metadata"] is not None
        assert result["metadata"]["input_templates_count"] is None
