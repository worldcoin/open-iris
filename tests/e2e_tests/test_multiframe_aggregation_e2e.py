import numpy as np
import pytest

import iris.io.errors as E
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import IrisTemplate
from iris.nodes.templates_aggregation.majority_vote import MajorityVoteAggregation
from iris.nodes.validators.object_validators import AreTemplatesAggregationCompatible
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import build_simple_multiframe_aggregation_output
from iris.pipelines.multiframe_aggregation_pipeline import MultiframeAggregationPipeline


@pytest.fixture
def sample_iris_templates():
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
def custom_aggregation_config():
    """Create a custom configuration for the aggregation pipeline."""
    return {
        "metadata": {"pipeline_name": "e2e_test_aggregation", "iris_version": "1.6.1"},
        "pipeline": [
            {
                "name": "templates_aggregation",
                "algorithm": {
                    "class_name": "iris.nodes.templates_aggregation.majority_vote.MajorityVoteAggregation",
                    "params": {
                        "consistency_threshold": 0.6,
                        "mask_threshold": 0.3,
                        "use_fragile_bits": True,
                        "fragile_bit_threshold": 0.2,
                    },
                },
                "inputs": [{"name": "templates", "source_node": "input"}],
                "callbacks": [
                    {
                        "class_name": "iris.nodes.validators.object_validators.AreTemplatesAggregationCompatible",
                        "params": {},
                    }
                ],
            }
        ],
    }


class TestMultiframeAggregationE2E:
    """End-to-end tests for multiframe aggregation functionality."""

    def test_full_pipeline_with_compatible_templates(self, sample_iris_templates, custom_aggregation_config):
        """Test the complete pipeline flow with compatible templates."""
        # Initialize the pipeline
        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config, subconfig_key="")

        # Run the pipeline
        result = pipeline.run(sample_iris_templates)

        # Verify the output structure
        assert isinstance(result, dict)
        assert "iris_template" in result
        assert "weights" in result
        assert "error" in result
        assert "metadata" in result

        # Verify successful execution (no error)
        assert result["error"] is None

        # Verify the aggregated template
        aggregated_template = result["iris_template"]
        assert isinstance(aggregated_template, IrisTemplate)
        assert aggregated_template.iris_code_version == sample_iris_templates[0].iris_code_version
        assert len(aggregated_template.iris_codes) == len(sample_iris_templates[0].iris_codes)
        assert len(aggregated_template.mask_codes) == len(sample_iris_templates[0].mask_codes)

        # Verify weights
        weights = result["weights"]
        assert isinstance(weights, list)
        assert len(weights) == len(sample_iris_templates[0].iris_codes)

        # Verify metadata
        metadata = result["metadata"]
        assert isinstance(metadata, dict)
        assert "templates_count" in metadata
        assert metadata["templates_count"] == len(sample_iris_templates)
        assert "iris_version" in metadata

    def test_pipeline_with_incompatible_templates(self, incompatible_iris_templates, custom_aggregation_config):
        """Test the pipeline with incompatible templates (should fail validation)."""
        # Initialize the pipeline
        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config, subconfig_key="")

        # Run the pipeline - should handle the validation error
        result = pipeline.run(incompatible_iris_templates)

        # Verify that an error was captured
        assert result["error"] is not None
        assert result["iris_template"] is None
        assert result["weights"] is None

    def test_pipeline_with_single_template(self, custom_aggregation_config):
        """Test the pipeline with a single template."""
        # Create a single template
        iris_codes = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
        mask_codes = [np.random.choice(2, size=(8, 32, 2)).astype(bool) for _ in range(2)]
        single_template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")

        # Initialize the pipeline
        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config, subconfig_key="")

        # Run the pipeline
        result = pipeline.run([single_template])

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None
        assert result["weights"] is not None

        # For single template, the output should be essentially the same as input
        aggregated_template = result["iris_template"]
        assert aggregated_template.iris_code_version == single_template.iris_code_version
        assert len(aggregated_template.iris_codes) == len(single_template.iris_codes)

    def test_pipeline_with_default_configuration(self, sample_iris_templates):
        """Test the pipeline using default configuration."""
        # Initialize with default config
        pipeline = MultiframeAggregationPipeline()

        # Run the pipeline
        result = pipeline.run(sample_iris_templates)

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None
        assert result["weights"] is not None

    def test_majority_vote_aggregation_standalone(self, sample_iris_templates):
        """Test MajorityVoteAggregation as a standalone component."""
        # Initialize the aggregation algorithm
        aggregator = MajorityVoteAggregation(
            consistency_threshold=0.7, mask_threshold=0.5, use_fragile_bits=True, fragile_bit_threshold=0.3
        )

        # Run aggregation
        combined_template, weights = aggregator.run(sample_iris_templates)

        # Verify output
        assert isinstance(combined_template, IrisTemplate)
        assert isinstance(weights, list)
        assert len(weights) == len(sample_iris_templates[0].iris_codes)

        # Verify template structure
        assert combined_template.iris_code_version == sample_iris_templates[0].iris_code_version
        assert len(combined_template.iris_codes) == len(sample_iris_templates[0].iris_codes)
        assert len(combined_template.mask_codes) == len(sample_iris_templates[0].mask_codes)

    def test_template_compatibility_validator_standalone(self, sample_iris_templates, incompatible_iris_templates):
        """Test AreTemplatesAggregationCompatible as a standalone component."""
        validator = AreTemplatesAggregationCompatible()

        # Test with compatible templates (should not raise)
        try:
            validator.run(sample_iris_templates)
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("Compatible templates should not raise an exception")

        # Test with incompatible templates (should raise)
        with pytest.raises(E.TemplateAggregationCompatibilityError):
            validator.run(incompatible_iris_templates)

    def test_pipeline_with_custom_environment(self, sample_iris_templates, custom_aggregation_config):
        """Test the pipeline with a custom environment."""
        # Create custom environment
        custom_env = Environment(
            pipeline_output_builder=build_simple_multiframe_aggregation_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        )

        # Initialize pipeline with custom environment
        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config, env=custom_env)

        # Run the pipeline
        result = pipeline.run(sample_iris_templates)

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None

    def test_pipeline_with_orb_environment(self, sample_iris_templates):
        """Test the pipeline with ORB environment."""
        # Use the predefined ORB environment
        pipeline = MultiframeAggregationPipeline(env=MultiframeAggregationPipeline.ORB_ENVIRONMENT)

        # Run the pipeline
        result = pipeline.run(sample_iris_templates)

        # Verify successful execution
        assert result["error"] is None
        assert result["iris_template"] is not None

        # ORB environment should serialize the template
        if result["iris_template"] is not None:
            # The ORB output builder might serialize the template differently
            pass  # Just verify it doesn't crash

    def test_pipeline_call_trace_functionality(self, sample_iris_templates, custom_aggregation_config):
        """Test that the pipeline call trace works correctly."""
        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config)

        # Run the pipeline
        _ = pipeline.run(sample_iris_templates)

        # Verify call trace has been populated
        assert pipeline.call_trace is not None

        # Check that input was written to call trace
        input_data = pipeline.call_trace.get_input()
        assert input_data == sample_iris_templates

        # Check that the aggregation node result is available
        aggregation_result = pipeline.call_trace["templates_aggregation"]
        assert aggregation_result is not None

    def test_pipeline_with_empty_templates_list(self, custom_aggregation_config):
        """Test the pipeline with an empty templates list."""
        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config)

        # Run with empty list - should be handled by validation
        result = pipeline.run([])

        # Should have an error due to validation failure
        assert result["error"] is not None
        assert result["iris_template"] is None

    def test_integration_with_different_template_sizes(self, custom_aggregation_config):
        """Test integration with templates of different sizes."""
        # Create templates with different but compatible sizes
        templates = []

        # All templates have same structure but different content
        for i in range(3):
            iris_codes = [np.random.choice(2, size=(4, 16, 2)).astype(bool) for _ in range(1)]  # Smaller size
            mask_codes = [np.random.choice(2, size=(4, 16, 2)).astype(bool) for _ in range(1)]
            template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
            templates.append(template)

        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config)
        result = pipeline.run(templates)

        # Should work fine with smaller templates
        assert result["error"] is None
        assert result["iris_template"] is not None

    def test_integration_with_many_templates(self, custom_aggregation_config):
        """Test integration with a large number of templates."""
        # Create many compatible templates
        templates = []
        for i in range(20):  # Large number of templates
            iris_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool) for _ in range(1)]
            mask_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool) for _ in range(1)]
            template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
            templates.append(template)

        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config)
        result = pipeline.run(templates)

        # Should handle many templates
        assert result["error"] is None
        assert result["iris_template"] is not None
        assert result["metadata"]["templates_count"] == 20

    def test_pipeline_parameter_propagation(self):
        """Test that parameters are correctly propagated through the pipeline."""
        # Create config with specific parameters
        config = {
            "metadata": {"pipeline_name": "param_test", "iris_version": "1.6.1"},
            "pipeline": [
                {
                    "name": "templates_aggregation",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_aggregation.majority_vote.MajorityVoteAggregation",
                        "params": {
                            "consistency_threshold": 0.9,  # High threshold
                            "mask_threshold": 0.1,  # Low threshold
                            "use_fragile_bits": False,  # Disabled
                            "fragile_bit_threshold": 0.0,
                        },
                    },
                    "inputs": [{"name": "templates", "source_node": "input"}],
                    "callbacks": [
                        {
                            "class_name": "iris.nodes.validators.object_validators.AreTemplatesAggregationCompatible",
                            "params": {},
                        }
                    ],
                }
            ],
        }

        # Create test templates
        templates = []
        for i in range(3):
            iris_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool)]
            mask_codes = [np.random.choice(2, size=(4, 8, 2)).astype(bool)]
            template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
            templates.append(template)

        pipeline = MultiframeAggregationPipeline(config=config)
        result = pipeline.run(templates)

        # Should work with custom parameters
        assert result["error"] is None
        assert result["iris_template"] is not None

    def test_error_handling_and_recovery(self, custom_aggregation_config):
        """Test error handling in the pipeline."""
        pipeline = MultiframeAggregationPipeline(config=custom_aggregation_config)

        # Test with None input (should be handled gracefully)
        try:
            result = pipeline.run(None)
            # Should either work or produce a meaningful error
            assert "error" in result
        except Exception as e:
            # If it raises an exception, it should be a meaningful one
            assert isinstance(e, (TypeError, AttributeError, ValueError))

    def test_pipeline_reproducibility(self, custom_aggregation_config):
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
        pipeline1 = MultiframeAggregationPipeline(config=custom_aggregation_config)
        result1 = pipeline1.run(templates)

        pipeline2 = MultiframeAggregationPipeline(config=custom_aggregation_config)
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
