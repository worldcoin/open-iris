from unittest.mock import Mock

import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import OutputFieldSpec
from iris.orchestration.output_builders import (
    _build_from_spec,
    build_iris_pipeline_orb_output,
    build_simple_iris_pipeline_debugging_output,
    build_simple_iris_pipeline_orb_output,
)


class TestOutputBuildersWithMissingKeys:
    """Test output builders behavior when keys are missing from call_trace."""

    @pytest.fixture
    def mock_call_trace_with_missing_keys(self):
        """Create a mock call_trace that has some keys missing."""
        # Create a call_trace with minimal required keys
        call_trace = PipelineCallTraceStorage(results_names=["encoder"])

        # Mock the input to avoid errors in metadata extraction
        mock_input = Mock()
        mock_input.width = 640
        mock_input.height = 480
        mock_input.eye_side = "left"
        call_trace.write_input(mock_input)

        # Deliberately don't write values for most keys, so they remain None
        # Don't write anything to encoder either, so it remains None
        # This will test the case where the key exists in storage but has None value

        return call_trace

    @pytest.fixture
    def mock_call_trace_with_simple_value(self):
        """Create a mock call_trace with a simple value for testing simple output builders."""
        # Create a call_trace with minimal required keys
        call_trace = PipelineCallTraceStorage(results_names=["encoder"])

        # Mock the input to avoid errors in metadata extraction
        mock_input = Mock()
        mock_input.width = 640
        mock_input.height = 480
        mock_input.eye_side = "left"
        call_trace.write_input(mock_input)

        # Write a simple value for encoder (for simple output builders that don't serialize)
        call_trace.write("encoder", "mock_iris_template")

        return call_trace

    def test_build_iris_pipeline_orb_output_with_missing_keys(self, mock_call_trace_with_missing_keys):
        """Test that build_iris_pipeline_orb_output includes None values for missing keys."""
        result = build_iris_pipeline_orb_output(mock_call_trace_with_missing_keys)

        # Check that the output contains the expected keys
        assert "error" in result
        assert "iris_template" in result
        assert "metadata" in result

        # Check that error is None (since no error was set)
        assert result["error"] is None

        # Check that iris_template is None (since we didn't write anything to encoder)
        assert result["iris_template"] is None

        # Check that metadata contains None values for missing keys
        metadata = result["metadata"]
        assert metadata["eye_centers"] is None
        assert metadata["pupil_to_iris_property"] is None
        assert metadata["offgaze_score"] is None
        assert metadata["eye_orientation"] is None
        assert metadata["occlusion90"] is None
        assert metadata["occlusion30"] is None
        assert metadata["iris_bbox"] is None
        assert metadata["sharpness_score"] is None

    def test_build_simple_iris_pipeline_orb_output_with_missing_keys(self, mock_call_trace_with_simple_value):
        """Test that build_simple_iris_pipeline_orb_output includes None values for missing keys."""
        result = build_simple_iris_pipeline_orb_output(mock_call_trace_with_simple_value)

        # Check that the output contains the expected keys
        assert "error" in result
        assert "iris_template" in result
        assert "metadata" in result

        # Check that error is None (since no error was set)
        assert result["error"] is None

        # Check that iris_template has the value we set
        assert result["iris_template"] == "mock_iris_template"

    def test_build_simple_iris_pipeline_debugging_output_with_missing_keys(self, mock_call_trace_with_missing_keys):
        """Test that build_simple_iris_pipeline_debugging_output includes None values for missing keys."""
        result = build_simple_iris_pipeline_debugging_output(mock_call_trace_with_missing_keys)

        # Check that keys with missing values are None
        assert result["extrapolated_polygons"] is None
        assert result["normalized_iris"] is None
        assert result["iris_response"] is None
        assert result["iris_response_refined"] is None
        assert result["error"] is None

    def test_custom_spec_with_missing_key(self, mock_call_trace_with_simple_value):
        """Test that _build_from_spec includes None for missing keys in custom specs."""
        # Create a custom spec that tries to extract a key that doesn't exist
        custom_spec = [
            OutputFieldSpec(key="existing_key", extractor=lambda ct: ct.get("encoder"), safe_serialize=False),
            OutputFieldSpec(key="missing_key", extractor=lambda ct: ct.get("nonexistent_key"), safe_serialize=False),
            OutputFieldSpec(
                key="another_missing_key", extractor=lambda ct: ct.get("another_nonexistent_key"), safe_serialize=True
            ),
        ]

        result = _build_from_spec(mock_call_trace_with_simple_value, custom_spec)

        # Check that existing key has the expected value
        assert result["existing_key"] == "mock_iris_template"

        # Check that missing keys have None values
        assert result["missing_key"] is None
        assert result["another_missing_key"] is None

        # Verify all expected keys are present
        assert set(result.keys()) == {"existing_key", "missing_key", "another_missing_key"}

    def test_missing_keys_result_in_key_none_pairs_in_output(self):
        """
        Explicit test showing that if a key is not in the call_trace,
        the output will contain key: None in it.
        """
        # Create a call_trace with no keys written (all will be None)
        call_trace = PipelineCallTraceStorage(results_names=[])

        # Mock the input to avoid errors in metadata extraction
        mock_input = Mock()
        mock_input.width = 640
        mock_input.height = 480
        mock_input.eye_side = "left"
        call_trace.write_input(mock_input)

        # Create a custom spec that tries to extract keys that don't exist
        custom_spec = [
            OutputFieldSpec(
                key="missing_key_1", extractor=lambda ct: ct.get("nonexistent_key_1"), safe_serialize=False
            ),
            OutputFieldSpec(
                key="missing_key_2", extractor=lambda ct: ct.get("nonexistent_key_2"), safe_serialize=False
            ),
            OutputFieldSpec(key="missing_key_3", extractor=lambda ct: ct.get("nonexistent_key_3"), safe_serialize=True),
        ]

        result = _build_from_spec(call_trace, custom_spec)

        # Verify that the output contains the keys with None values
        expected_output = {
            "missing_key_1": None,
            "missing_key_2": None,
            "missing_key_3": None,
        }

        assert result == expected_output

        # Explicitly verify each key: None pair
        assert result["missing_key_1"] is None
        assert result["missing_key_2"] is None
        assert result["missing_key_3"] is None

        # Verify all keys are present in the output
        assert "missing_key_1" in result
        assert "missing_key_2" in result
        assert "missing_key_3" in result
