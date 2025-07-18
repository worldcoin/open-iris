from unittest.mock import Mock, patch

import numpy as np
import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import IrisTemplate
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import build_simple_multiframe_aggregation_output
from iris.pipelines.multiframe_aggregation_pipeline import MultiframeAggregationPipeline


@pytest.fixture
def mock_iris_template():
    """Create a mock IrisTemplate for testing."""
    iris_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
    mask_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
    return IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")


@pytest.fixture
def mock_templates_list(mock_iris_template):
    """Create a list of mock IrisTemplates for testing."""
    return [mock_iris_template for _ in range(3)]


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return {
        "metadata": {"pipeline_name": "test_aggregation", "iris_version": "1.6.1"},
        "pipeline": [
            {
                "name": "templates_aggregation",
                "algorithm": {
                    "class_name": "iris.nodes.templates_aggregation.majority_vote.MajorityVoteAggregation",
                    "params": {
                        "consistency_threshold": 0.75,
                        "mask_threshold": 0.5,
                        "use_inconsistent_bits": True,
                        "inconsistent_bit_threshold": 0.4,
                    },
                },
                "inputs": [{"name": "templates", "source_node": "input"}],
                "callbacks": [],
            }
        ],
    }


@pytest.fixture
def mock_full_config():
    """Mock configuration that contains both: IrisPipeline config and MultiframeAggregationPipeline config"""
    return {
        "metadata": {"pipeline_name": "iris_pipeline", "iris_version": "1.6.1"},
        "pipeline": [{"name": "iris_pipeline", "algorithm": {}}],
        "templates_aggregation": {
            "metadata": {"pipeline_name": "templates_aggregation", "iris_version": "1.6.1"},
            "pipeline": [
                {
                    "name": "templates_aggregation",
                    "algorithm": {
                        "class_name": "iris.nodes.templates_aggregation.majority_vote.MajorityVoteAggregation",
                        "params": {
                            "consistency_threshold": 0.75,
                            "mask_threshold": 0.5,
                            "use_inconsistent_bits": True,
                            "inconsistent_bit_threshold": 0.4,
                        },
                    },
                    "inputs": [{"name": "templates", "source_node": "input"}],
                    "callbacks": [],
                }
            ],
        },
    }


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    return Environment(
        pipeline_output_builder=build_simple_multiframe_aggregation_output,
        error_manager=store_error_manager,
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )


class TestMultiframeAggregationPipeline:
    """Test suite for MultiframeAggregationPipeline class."""

    def test_init_with_dict_config(self, mock_config, mock_environment):
        """Test initialization with dictionary configuration."""
        pipeline = MultiframeAggregationPipeline(config=mock_config, env=mock_environment, subconfig_key="")

        assert pipeline.params.metadata.pipeline_name == "test_aggregation"
        assert pipeline.params.metadata.iris_version == "1.6.1"
        assert len(pipeline.params.pipeline) == 1
        assert pipeline.env == mock_environment

    def test_init_with_full_config(self, mock_full_config, mock_environment):
        """Test initialization with full configuration."""
        pipeline = MultiframeAggregationPipeline(
            config=mock_full_config, env=mock_environment, subconfig_key="templates_aggregation"
        )

        assert pipeline.params.metadata.pipeline_name == "templates_aggregation"
        assert pipeline.params.metadata.iris_version == "1.6.1"
        assert len(pipeline.params.pipeline) == 1

        # default key
        pipeline = MultiframeAggregationPipeline(config=mock_full_config, env=mock_environment)

        assert pipeline.params.metadata.pipeline_name == "templates_aggregation"
        assert pipeline.params.metadata.iris_version == "1.6.1"
        assert len(pipeline.params.pipeline) == 1

        # invalid key should raise ValueError
        with pytest.raises(ValueError):
            MultiframeAggregationPipeline(config=mock_full_config, env=mock_environment, subconfig_key="invalid_key")

    def test_init_with_none_config(self):
        """Test initialization with None configuration (should load default)."""
        with patch.object(MultiframeAggregationPipeline, "load_config") as mock_load:
            mock_load.return_value = {"metadata": {"pipeline_name": "default", "iris_version": "1.6.1"}, "pipeline": []}

            _ = MultiframeAggregationPipeline(config=None)
            mock_load.assert_called_once_with(None, keyword="templates_aggregation")

    def test_init_with_string_config(self):
        """Test initialization with string configuration."""
        yaml_config = """
        metadata:
          pipeline_name: "yaml_test"
          iris_version: "1.6.1"
        pipeline: []
        """

        with patch.object(MultiframeAggregationPipeline, "load_config") as mock_load:
            mock_load.return_value = {
                "metadata": {"pipeline_name": "yaml_test", "iris_version": "1.6.1"},
                "pipeline": [],
            }

            _ = MultiframeAggregationPipeline(config=yaml_config)
            mock_load.assert_called_once_with(yaml_config, keyword="templates_aggregation")

    def test_init_with_custom_subconfig_key(self, mock_config):
        """Test initialization with custom subconfig key."""
        with patch.object(MultiframeAggregationPipeline, "load_config") as mock_load:
            mock_load.return_value = mock_config

            _ = MultiframeAggregationPipeline(config="test", subconfig_key="custom_key")
            mock_load.assert_called_once_with("test", keyword="custom_key")

    def test_run_with_templates(self, mock_config, mock_templates_list):
        """Test running the pipeline with templates."""
        with patch("iris.pipelines.multiframe_aggregation_pipeline.BasePipeline.run") as mock_base_run:
            mock_base_run.return_value = {"result": "success"}

            pipeline = MultiframeAggregationPipeline(config=mock_config, subconfig_key="")
            _ = pipeline.run(mock_templates_list)

            # Verify that BasePipeline.run was called with correct input format
            mock_base_run.assert_called_once()
            call_args = mock_base_run.call_args[0]
            assert call_args[0] == {"templates": mock_templates_list}

    def test_handle_input(self, mock_config, mock_templates_list):
        """Test _handle_input method."""
        pipeline = MultiframeAggregationPipeline(config=mock_config, subconfig_key="")
        pipeline.call_trace = Mock()

        pipeline_input = {"templates": mock_templates_list}
        pipeline._handle_input(pipeline_input)

        pipeline.call_trace.write_input.assert_called_once_with(mock_templates_list)

    def test_handle_output(self, mock_config):
        """Test _handle_output method."""
        # Create a mock output builder function
        mock_output = {"iris_template": None, "weights": None, "error": None, "metadata": {}}
        mock_output_builder = Mock(return_value=mock_output)

        # Create environment with mock output builder
        mock_env = Environment(
            pipeline_output_builder=mock_output_builder,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        )

        pipeline = MultiframeAggregationPipeline(config=mock_config, env=mock_env, subconfig_key="")
        pipeline.call_trace = Mock()

        result = pipeline._handle_output()

        mock_output_builder.assert_called_once_with(pipeline.call_trace)
        assert result == mock_output

    def test_load_config_with_none(self):
        """Test load_config static method with None input."""
        with patch("iris.pipelines.multiframe_aggregation_pipeline.load_yaml_config") as mock_load_yaml:
            mock_load_yaml.return_value = {"templates_aggregation": {"metadata": {"pipeline_name": "test"}}}

            result = MultiframeAggregationPipeline.load_config(None)

            # Verify load_yaml_config was called
            mock_load_yaml.assert_called_once()
            assert result == {"metadata": {"pipeline_name": "test"}}

    def test_load_config_with_yaml_string(self):
        """Test load_config static method with YAML string."""
        yaml_string = "templates_aggregation:\n  metadata:\n    pipeline_name: test"

        with patch("iris.pipelines.multiframe_aggregation_pipeline.load_yaml_config") as mock_load_yaml:
            mock_load_yaml.return_value = {"templates_aggregation": {"metadata": {"pipeline_name": "test"}}}

            _ = MultiframeAggregationPipeline.load_config(yaml_string)

            # Just verify it was called with the yaml string and some path
            mock_load_yaml.assert_called_once()
            call_args = mock_load_yaml.call_args[0]
            assert call_args[0] == yaml_string

    def test_load_config_with_empty_keyword(self):
        """Test load_config with empty keyword (standalone config)."""
        with patch("iris.pipelines.multiframe_aggregation_pipeline.load_yaml_config") as mock_load_yaml:
            standalone_config = {"metadata": {"pipeline_name": "standalone"}}
            mock_load_yaml.return_value = standalone_config

            result = MultiframeAggregationPipeline.load_config(None, keyword="")

            assert result == standalone_config

    def test_load_config_missing_keyword(self):
        """Test load_config with missing keyword raises ValueError."""
        with patch("iris.pipelines.multiframe_aggregation_pipeline.load_yaml_config") as mock_load_yaml:
            mock_load_yaml.return_value = {"other_key": {"data": "value"}}

            with pytest.raises(ValueError):
                MultiframeAggregationPipeline.load_config(None, keyword="missing_key")

    def test_load_config_custom_keyword(self):
        """Test load_config with custom keyword."""
        with patch("iris.pipelines.multiframe_aggregation_pipeline.load_yaml_config") as mock_load_yaml:
            mock_load_yaml.return_value = {"custom_aggregation": {"metadata": {"pipeline_name": "custom"}}}

            result = MultiframeAggregationPipeline.load_config(None, keyword="custom_aggregation")

            assert result == {"metadata": {"pipeline_name": "custom"}}

    def test_load_from_config_success(self):
        """Test load_from_config class method with successful loading."""
        base64_config = "dGVzdF9jb25maWc="  # base64 encoded "test_config"

        with patch("iris.pipelines.multiframe_aggregation_pipeline.base64_decode_str") as mock_decode:
            mock_decode.return_value = "decoded_config"

            with patch.object(MultiframeAggregationPipeline, "__init__", return_value=None):
                mock_pipeline = Mock()

                with patch.object(MultiframeAggregationPipeline, "__new__", return_value=mock_pipeline):
                    result = MultiframeAggregationPipeline.load_from_config(base64_config)

                assert result["agent"] == mock_pipeline
                assert result["error"] is None
                mock_decode.assert_called_once_with(base64_config)

    def test_load_from_config_with_exception(self):
        """Test load_from_config class method with exception handling."""
        base64_config = "invalid_base64"

        with patch("iris.pipelines.multiframe_aggregation_pipeline.base64_decode_str") as mock_decode:
            mock_decode.side_effect = ValueError("Invalid base64")

            result = MultiframeAggregationPipeline.load_from_config(base64_config)

            assert result["agent"] is None
            assert result["error"] is not None
            assert result["error"]["error_type"] == "ValueError"
            assert result["error"]["message"] == "Invalid base64"
            assert "traceback" in result["error"]

    def test_orb_environment_class_attribute(self):
        """Test that ORB_ENVIRONMENT class attribute is properly defined."""
        assert hasattr(MultiframeAggregationPipeline, "ORB_ENVIRONMENT")
        orb_env = MultiframeAggregationPipeline.ORB_ENVIRONMENT

        assert hasattr(orb_env, "pipeline_output_builder")
        assert hasattr(orb_env, "error_manager")
        assert hasattr(orb_env, "call_trace_initialiser")

    def test_parameters_class(self):
        """Test the Parameters inner class."""
        params_data = {"metadata": {"pipeline_name": "test", "iris_version": "1.6.1"}, "pipeline": []}

        params = MultiframeAggregationPipeline.Parameters(**params_data)

        assert params.metadata.pipeline_name == "test"
        assert params.metadata.iris_version == "1.6.1"
        assert params.pipeline == []

    def test_parameters_type_attribute(self):
        """Test that __parameters_type__ is correctly set."""
        assert MultiframeAggregationPipeline.__parameters_type__ == MultiframeAggregationPipeline.Parameters

    @patch("iris.pipelines.multiframe_aggregation_pipeline.BasePipeline.__init__")
    def test_inheritance_from_base_pipeline(self, mock_base_init, mock_config):
        """Test that MultiframeAggregationPipeline properly inherits from BasePipeline."""
        mock_env = Mock()

        pipeline = MultiframeAggregationPipeline.__new__(MultiframeAggregationPipeline)
        pipeline.__init__(config=mock_config, env=mock_env, subconfig_key="")

        # Verify that BasePipeline.__init__ was called with correct arguments
        mock_base_init.assert_called_once_with(mock_config, mock_env)

    def test_run_with_additional_args_kwargs(self, mock_templates_list):
        """Test running the pipeline with additional args and kwargs."""
        with patch("iris.pipelines.multiframe_aggregation_pipeline.BasePipeline.run") as mock_base_run:
            mock_base_run.return_value = {"result": "success"}

            with patch.object(MultiframeAggregationPipeline, "load_config") as mock_load:
                mock_load.return_value = {
                    "metadata": {"pipeline_name": "test", "iris_version": "1.6.1"},
                    "pipeline": [],
                }

                pipeline = MultiframeAggregationPipeline()
                _ = pipeline.run(mock_templates_list, "extra_arg", extra_kwarg="value")

                # Verify that additional args and kwargs are passed through
                mock_base_run.assert_called_once()
                call_args, call_kwargs = mock_base_run.call_args
                assert call_args[1] == "extra_arg"
                assert call_kwargs["extra_kwarg"] == "value"

    def test_handle_input_with_additional_args_kwargs(self, mock_templates_list):
        """Test _handle_input method with additional args and kwargs."""
        with patch.object(MultiframeAggregationPipeline, "load_config") as mock_load:
            mock_load.return_value = {"metadata": {"pipeline_name": "test", "iris_version": "1.6.1"}, "pipeline": []}

            pipeline = MultiframeAggregationPipeline()
            pipeline.call_trace = Mock()

            pipeline_input = {"templates": mock_templates_list}
            pipeline._handle_input(pipeline_input, "extra_arg", extra_kwarg="value")

            # Should still write the templates to call trace regardless of extra args
            pipeline.call_trace.write_input.assert_called_once_with(mock_templates_list)
