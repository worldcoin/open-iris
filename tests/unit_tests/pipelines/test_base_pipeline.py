"""
Unit tests for BasePipeline class.

This module contains tests that verify the behavior of the BasePipeline class,
specifically focusing on the equivalence between the `estimate()` and `run()` methods.

The main test validates that:
1. The `estimate()` method produces identical output to the `run()` method
2. Both methods handle additional arguments and keyword arguments consistently
3. The `estimate()` method internally calls the `run()` method (as documented)
4. Multiple calls to both methods remain consistent

These tests ensure that the `estimate()` method correctly wraps the `run()` method
to match the Orb system AI models call interface, as stated in the docstring.
"""

from typing import Any, Dict, List
from unittest.mock import Mock

import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.pipeline_dataclasses import PipelineMetadata, PipelineNode
from iris.pipelines.base_pipeline import BasePipeline


def custom_test_output_builder(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Custom output builder for testing that handles dictionary inputs."""
    return {"input": call_trace.get_input(), "metadata": {"test": "output"}, "error": call_trace.get("error")}


class ConcretePipeline(BasePipeline[Dict[str, Any], Dict[str, Any]]):
    """Concrete implementation of BasePipeline for testing purposes."""

    PACKAGE_VERSION = "1.0.0"

    class Parameters(BasePipeline.Parameters):
        """Parameters class for ConcretePipeline."""

        metadata: PipelineMetadata
        pipeline: List[PipelineNode]

    __parameters_type__ = Parameters

    def _handle_input(self, pipeline_input: Dict[str, Any], *args: Any, **kwargs: Any) -> None:
        """Handle input by writing to call trace."""
        self.call_trace.write_input(pipeline_input)

    def _handle_output(self, *args, **kwargs) -> Dict[str, Any]:
        """Handle output by building from call trace."""
        return self.env.pipeline_output_builder(self.call_trace)


@pytest.fixture
def mock_environment():
    """Create a mock environment for testing."""
    return Environment(
        pipeline_output_builder=custom_test_output_builder,
        error_manager=store_error_manager,
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )


@pytest.fixture
def simple_config():
    """Create a simple pipeline configuration for testing."""
    return {"metadata": {"pipeline_name": "test_pipeline", "iris_version": "1.6.1"}, "pipeline": []}


@pytest.fixture
def test_input():
    """Create test input data."""
    return {"test_data": "sample_value", "number": 42}


class TestBasePipeline:
    """Test suite for BasePipeline class."""

    def test_estimate_and_run_produce_same_output(self, simple_config, mock_environment, test_input):
        """Test that estimate() and run() methods produce identical outputs."""
        # Create pipeline instance
        pipeline = ConcretePipeline(config=simple_config, env=mock_environment)

        # Run both methods with the same input
        estimate_result = pipeline.estimate(test_input)
        run_result = pipeline.run(test_input)

        # Assert that both methods produce the same output
        assert estimate_result == run_result

    def test_estimate_and_run_with_args_kwargs(self, simple_config, mock_environment, test_input):
        """Test that estimate() and run() produce same output with additional args and kwargs."""
        # Create pipeline instance
        pipeline = ConcretePipeline(config=simple_config, env=mock_environment)

        # Test with additional positional and keyword arguments
        extra_arg = "extra_positional"
        extra_kwarg = {"extra": "keyword"}

        # Run both methods with the same input and additional arguments
        estimate_result = pipeline.estimate(test_input, extra_arg, extra_param=extra_kwarg)
        run_result = pipeline.run(test_input, extra_arg, extra_param=extra_kwarg)

        # Assert that both methods produce the same output
        assert estimate_result == run_result

    def test_estimate_calls_run_internally(self, simple_config, mock_environment, test_input):
        """Test that estimate() method internally calls run() method."""
        # Create pipeline instance
        pipeline = ConcretePipeline(config=simple_config, env=mock_environment)

        # Mock the run method to verify it's called by estimate
        original_run = pipeline.run
        pipeline.run = Mock(return_value={"mocked": "result"})

        # Call estimate
        result = pipeline.estimate(test_input, "arg1", kwarg1="value1")

        # Verify run was called with the same arguments
        pipeline.run.assert_called_once_with(test_input, "arg1", kwarg1="value1")
        assert result == {"mocked": "result"}

        # Restore original run method
        pipeline.run = original_run

    def test_estimate_and_run_multiple_calls_consistency(self, simple_config, mock_environment, test_input):
        """Test that multiple calls to estimate() and run() remain consistent."""
        # Create pipeline instance
        pipeline = ConcretePipeline(config=simple_config, env=mock_environment)

        # Make multiple calls and verify consistency
        for i in range(3):
            estimate_result = pipeline.estimate(test_input)
            run_result = pipeline.run(test_input)

            assert estimate_result == run_result, f"Mismatch on iteration {i}"

    def test_valid_package_version_passes(self):
        """Test that a valid PACKAGE_VERSION allows class creation."""

        class ValidPipeline(BasePipeline):
            PACKAGE_VERSION = "1.0.0"

            def _handle_input(self, pipeline_input, *args, **kwargs):
                pass

            def _handle_output(self, *args, **kwargs):
                return None

        # Should not raise any exception
        assert ValidPipeline.PACKAGE_VERSION == "1.0.0"

    def test_missing_package_version_raises_error(self):
        """Test that missing PACKAGE_VERSION raises TypeError on class creation."""

        with pytest.raises(TypeError) as exc_info:

            class InvalidPipeline(BasePipeline):
                # Missing PACKAGE_VERSION
                def _handle_input(self, pipeline_input, *args, **kwargs):
                    pass

                def _handle_output(self, *args, **kwargs):
                    return None

        error_message = str(exc_info.value)
        assert "InvalidPipeline must define a non-empty string PACKAGE_VERSION class attribute" in error_message

    def test_empty_string_package_version_raises_error(self):
        """Test that empty string PACKAGE_VERSION raises TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class InvalidPipeline(BasePipeline):
                PACKAGE_VERSION = ""  # Empty string

                def _handle_input(self, pipeline_input, *args, **kwargs):
                    pass

                def _handle_output(self, *args, **kwargs):
                    return None

        error_message = str(exc_info.value)
        assert "InvalidPipeline must define a non-empty string PACKAGE_VERSION class attribute" in error_message

    def test_none_package_version_raises_error(self):
        """Test that None PACKAGE_VERSION raises TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class InvalidPipeline(BasePipeline):
                PACKAGE_VERSION = None

                def _handle_input(self, pipeline_input, *args, **kwargs):
                    pass

                def _handle_output(self, *args, **kwargs):
                    return None

        error_message = str(exc_info.value)
        assert "InvalidPipeline must define a non-empty string PACKAGE_VERSION class attribute" in error_message

    @pytest.mark.parametrize(
        "invalid_version",
        [
            123,
            None,
            [],
            {},
            object(),
        ],
        ids=[
            "integer",
            "none",
            "list",
            "dict",
            "object",
        ],
    )
    def test_various_invalid_package_versions_raise_error(self, invalid_version):
        """Test that various invalid PACKAGE_VERSION types raise TypeError."""

        with pytest.raises(TypeError) as exc_info:

            class InvalidPipeline(BasePipeline):
                PACKAGE_VERSION = invalid_version

                def _handle_input(self, pipeline_input, *args, **kwargs):
                    pass

                def _handle_output(self, *args, **kwargs):
                    return None

        error_message = str(exc_info.value)
        assert "InvalidPipeline must define a non-empty string PACKAGE_VERSION class attribute" in error_message
