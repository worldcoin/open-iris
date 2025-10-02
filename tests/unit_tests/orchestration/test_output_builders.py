from unittest.mock import Mock

import numpy as np
import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import (
    AlignedTemplates,
    DistanceMatrix,
    EyeCenters,
    IrisTemplateWithId,
    Offgaze,
    OutputFieldSpec,
    WeightedIrisTemplate,
)
from iris.orchestration import output_builders as ob
from iris.orchestration.output_builders import (
    _build_from_spec,
    build_aggregation_templates_orb_output,
    build_iris_pipeline_orb_output,
    build_multiframe_iris_pipeline_orb_output,
    build_simple_iris_pipeline_debugging_output,
    build_simple_iris_pipeline_orb_output,
    build_simple_multiframe_iris_pipeline_output,
)

SAFE_SERIALIZE = getattr(ob, "__safe_serialize")


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


class TestBuildAggregationTemplatesOrbOutput:
    @pytest.fixture
    def mock_call_trace_with_missing_keys(self):
        call_trace = PipelineCallTraceStorage(results_names=["templates_aggregation"])
        # No input, no aggregation, no alignment, no identity filter
        return call_trace

    @pytest.fixture
    def mock_call_trace_with_aggregation(self):
        call_trace = PipelineCallTraceStorage(results_names=["templates_aggregation"])
        # Minimal iris template data
        iris_codes = [np.ones((2, 2, 2), dtype=bool)]
        mask_codes = [np.zeros((2, 2, 2), dtype=bool)]
        weights = [np.ones((2, 2, 2), dtype=np.float32)]
        iris_code_version = "v2.1"
        weighted_template = WeightedIrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            weights=weights,
            iris_code_version=iris_code_version,
        )
        templaets_with_ids = [
            IrisTemplateWithId(
                iris_codes=weighted_template.iris_codes,
                mask_codes=weighted_template.mask_codes,
                iris_code_version=weighted_template.iris_code_version,
                image_id=f"image_id_{i}",
            )
            for i in range(1)
        ]

        call_trace.write("templates_aggregation", weighted_template)
        # Set input as a list of templates (simulate input)
        call_trace.write_input(templaets_with_ids)
        return call_trace

    @pytest.fixture
    def mock_call_trace_with_alignment_and_identity_filter(self):
        call_trace = PipelineCallTraceStorage(
            results_names=["templates_aggregation", "templates_alignment", "identity_filter"]
        )
        # Minimal iris template data
        iris_codes = [np.ones((2, 2, 2), dtype=bool)]
        mask_codes = [np.zeros((2, 2, 2), dtype=bool)]
        weights = [np.ones((2, 2, 2), dtype=np.float32)]
        iris_code_version = "v2.1"
        weighted_template = WeightedIrisTemplate(
            iris_codes=iris_codes,
            mask_codes=mask_codes,
            weights=weights,
            iris_code_version=iris_code_version,
        )
        templaets_with_ids = [
            IrisTemplateWithId(
                iris_codes=weighted_template.iris_codes,
                mask_codes=weighted_template.mask_codes,
                iris_code_version=weighted_template.iris_code_version,
                image_id=f"image_id_{i}",
            )
            for i in range(1)
        ]
        call_trace.write("templates_aggregation", weighted_template)
        call_trace.write_input(templaets_with_ids)
        # Add aligned templates
        aligned_templates = AlignedTemplates(
            templates=templaets_with_ids,
            distances=DistanceMatrix(data={(0, 0): 0.0}),
            reference_template_id=0,
        )
        call_trace.write("templates_alignment", aligned_templates)
        # Add identity filter result
        call_trace.write("identity_validation", templaets_with_ids)
        return call_trace

    def test_with_missing_keys(self, mock_call_trace_with_missing_keys):
        result = build_aggregation_templates_orb_output(mock_call_trace_with_missing_keys)
        assert set(result.keys()) == {"error", "iris_template", "metadata"}
        assert result["error"] is None
        assert result["iris_template"] is None
        metadata = result["metadata"]
        assert metadata["input_templates_count"] is None
        assert metadata["aligned_templates"]["reference_template_id"] is None
        assert metadata["aligned_templates"]["distances"] is None
        assert metadata["post_identity_filter_templates_count"] is None

    def test_with_aggregation(self, mock_call_trace_with_aggregation):
        result = build_aggregation_templates_orb_output(mock_call_trace_with_aggregation)
        assert set(result.keys()) == {"error", "iris_template", "metadata"}
        assert result["error"] is None
        # Should be safely serialized (dict)
        assert isinstance(result["iris_template"], dict)
        metadata = result["metadata"]
        assert metadata["input_templates_count"] == 1
        assert metadata["aligned_templates"]["reference_template_id"] is None
        assert metadata["aligned_templates"]["distances"] is None
        assert metadata["post_identity_filter_templates_count"] is None

    def test_with_alignment_and_identity_filter(self, mock_call_trace_with_alignment_and_identity_filter):
        result = build_aggregation_templates_orb_output(mock_call_trace_with_alignment_and_identity_filter)
        assert set(result.keys()) == {"error", "iris_template", "metadata"}
        assert result["error"] is None
        assert isinstance(result["iris_template"], dict)
        metadata = result["metadata"]
        assert metadata["input_templates_count"] == 1
        assert metadata["aligned_templates"]["reference_template_id"] == 0
        assert metadata["aligned_templates"]["distances"] == {"0_0": 0.0}
        assert metadata["post_identity_filter_templates_count"] == 1


class TestSafeSerialize:
    def test_none_returns_none(self):
        assert SAFE_SERIALIZE(None) is None

    def test_immutable_model_serialization(self):
        offgaze = Offgaze(score=0.5)
        assert SAFE_SERIALIZE(offgaze) == offgaze.serialize()

        eye_centers = EyeCenters(pupil_x=1.0, pupil_y=2.0, iris_x=3.0, iris_y=4.0)
        assert SAFE_SERIALIZE(eye_centers) == eye_centers.serialize()

    def test_list_and_tuple_recursion(self):
        offgaze = Offgaze(score=0.25)
        arr = np.array([1, 2, 3])

        data_list = [1, "a", offgaze, arr]
        assert SAFE_SERIALIZE(data_list) == [1, "a", offgaze.serialize(), arr]

        data_tuple = (True, 3.14, offgaze, arr)
        assert SAFE_SERIALIZE(data_tuple) == (True, 3.14, offgaze.serialize(), arr)

    def test_primitives_passthrough(self):
        assert SAFE_SERIALIZE("hello") == "hello"
        assert SAFE_SERIALIZE(123) == 123
        assert SAFE_SERIALIZE(3.14) == 3.14
        assert SAFE_SERIALIZE(True) is True

    def test_unsupported_type_raises(self):
        class Foo:
            pass

        with pytest.raises(NotImplementedError):
            SAFE_SERIALIZE(Foo())


class TestOutputBuildersNoneValues:
    """Test output builders behavior when various keys are None or missing."""

    @pytest.fixture
    def mock_call_trace_aggregation_none(self):
        """Create a mock call_trace where aggregation_result is None."""
        call_trace = PipelineCallTraceStorage(results_names=["aggregation_result"])

        # Mock the input to avoid errors in metadata extraction
        mock_input = [Mock()]
        mock_input[0].eye_side = "left"
        call_trace.write_input(mock_input)

        # Explicitly set aggregation_result to None
        call_trace.write("aggregation_result", None)

        # Add empty individual_frames to avoid KeyError
        call_trace.write("individual_frames", [])

        return call_trace

    @pytest.fixture
    def mock_call_trace_all_missing(self):
        """Create a mock call_trace where all optional keys are missing."""
        call_trace = PipelineCallTraceStorage(results_names=[])

        # Mock the input to avoid errors in metadata extraction
        mock_input = [Mock()]
        mock_input[0].eye_side = "left"
        call_trace.write_input(mock_input)

        # Don't write any optional keys - they will all be None/missing

        return call_trace

    @pytest.fixture
    def mock_call_trace_empty_input(self):
        """Create a mock call_trace where input is empty."""
        call_trace = PipelineCallTraceStorage(results_names=["aggregation_result", "individual_frames"])

        # Set empty input
        call_trace.write_input([])

        # Set other keys to None/empty
        call_trace.write("aggregation_result", None)
        call_trace.write("individual_frames", [])

        return call_trace

    @pytest.fixture
    def mock_call_trace_none_input(self):
        """Create a mock call_trace where input is None."""
        call_trace = PipelineCallTraceStorage(results_names=["aggregation_result", "individual_frames"])

        # Set None input
        call_trace.write_input(None)

        # Set other keys to None/empty
        call_trace.write("aggregation_result", None)
        call_trace.write("individual_frames", [])

        return call_trace

    @pytest.fixture
    def mock_call_trace_aggregation_with_error(self):
        """Create a mock call_trace where aggregation_result contains an error."""
        call_trace = PipelineCallTraceStorage(results_names=["aggregation_result", "individual_frames"])

        # Mock the input to avoid errors in metadata extraction
        mock_input = [Mock()]
        mock_input[0].eye_side = "left"
        call_trace.write_input(mock_input)

        # Set aggregation_result with error
        aggregation_result = {
            "error": {
                "error_type": "IdentityValidationError",
                "message": "Found multiple identity clusters",
                "traceback": "traceback here...",
            },
            "iris_template": None,
            "metadata": {"some": "metadata"},
        }
        call_trace.write("aggregation_result", aggregation_result)

        # Add empty individual_frames to avoid KeyError
        call_trace.write("individual_frames", [])

        return call_trace

    def test_multiframe_orb_output_with_none_aggregation_result(self, mock_call_trace_aggregation_none):
        """Test multiframe ORB output builder when aggregation_result is None."""
        from iris.orchestration.output_builders import build_multiframe_iris_pipeline_orb_output

        output = build_multiframe_iris_pipeline_orb_output(mock_call_trace_aggregation_none)

        # Check that iris_template is None when aggregation_result is None
        assert output["iris_template"] is None

        # Check that metadata is still generated
        assert "metadata" in output
        assert output["metadata"]["iris_version"] is not None
        assert output["metadata"]["input_images_count"] == 1
        assert output["metadata"]["eye_side"] == "left"
        assert output["metadata"]["aggregation_successful"] is False
        assert output["metadata"]["is_aggregated"] is False

        # Check other fields
        assert "error" in output
        assert "individual_frames" in output
        assert output["individual_frames"] == []
        assert "templates_aggregation_metadata" in output
        assert output["templates_aggregation_metadata"] is None

    def test_multiframe_orb_output_with_all_keys_missing(self, mock_call_trace_all_missing):
        """Test multiframe ORB output builder when all optional keys are missing."""

        output = build_multiframe_iris_pipeline_orb_output(mock_call_trace_all_missing)

        # Check that iris_template is None when aggregation_result is missing
        assert output["iris_template"] is None

        # Check that metadata is still generated with available data
        assert "metadata" in output
        assert output["metadata"]["iris_version"] is not None
        assert output["metadata"]["input_images_count"] == 1
        assert output["metadata"]["eye_side"] == "left"
        assert output["metadata"]["aggregation_successful"] is False
        assert output["metadata"]["is_aggregated"] is False

        # Check other fields handle missing keys gracefully
        assert "error" in output
        assert output["error"] is None  # No error stored
        assert "individual_frames" in output
        assert output["individual_frames"] == []  # Default empty list
        assert "templates_aggregation_metadata" in output
        assert output["templates_aggregation_metadata"] is None

    def test_multiframe_orb_output_with_empty_input(self, mock_call_trace_empty_input):
        """Test multiframe ORB output builder when input is empty."""

        output = build_multiframe_iris_pipeline_orb_output(mock_call_trace_empty_input)

        # Check that iris_template is None
        assert output["iris_template"] is None

        # Check that metadata handles empty input gracefully
        assert "metadata" in output
        assert output["metadata"]["iris_version"] is not None
        assert output["metadata"]["input_images_count"] is None  # Empty input
        assert output["metadata"]["eye_side"] is None  # No first entry
        assert output["metadata"]["aggregation_successful"] is False
        assert output["metadata"]["is_aggregated"] is False

        # Check other fields
        assert "error" in output
        assert "individual_frames" in output
        assert output["individual_frames"] == []
        assert "templates_aggregation_metadata" in output
        assert output["templates_aggregation_metadata"] is None

    def test_multiframe_orb_output_with_none_input(self, mock_call_trace_none_input):
        """Test multiframe ORB output builder when input is None."""

        output = build_multiframe_iris_pipeline_orb_output(mock_call_trace_none_input)

        # Check that iris_template is None
        assert output["iris_template"] is None

        # Check that metadata handles None input gracefully
        assert "metadata" in output
        assert output["metadata"]["iris_version"] is not None
        assert output["metadata"]["input_images_count"] is None  # None input
        assert output["metadata"]["eye_side"] is None  # No first entry
        assert output["metadata"]["aggregation_successful"] is False
        assert output["metadata"]["is_aggregated"] is False

        # Check other fields
        assert "error" in output
        assert "individual_frames" in output
        assert output["individual_frames"] == []
        assert "templates_aggregation_metadata" in output
        assert output["templates_aggregation_metadata"] is None

    def test_multiframe_simple_output_with_none_values(self, mock_call_trace_all_missing):
        """Test multiframe simple output builder when all optional keys are missing."""

        output = build_simple_multiframe_iris_pipeline_output(mock_call_trace_all_missing)

        # Check same structure as ORB output but without serialization
        assert output["iris_template"] is None
        assert "metadata" in output
        assert output["metadata"]["aggregation_successful"] is False
        assert output["metadata"]["is_aggregated"] is False
        assert "error" in output
        assert output["error"] is None
        assert "individual_frames" in output
        assert output["individual_frames"] == []
        assert "templates_aggregation_metadata" in output
        assert output["templates_aggregation_metadata"] is None

    def test_multiframe_output_with_aggregation_error(self, mock_call_trace_aggregation_with_error):
        """Test multiframe output builder when aggregation_result contains an error."""
        from iris.orchestration.output_builders import build_multiframe_iris_pipeline_orb_output

        output = build_multiframe_iris_pipeline_orb_output(mock_call_trace_aggregation_with_error)

        # Check that iris_template is None when aggregation has error
        assert output["iris_template"] is None

        # Check that metadata reflects failed aggregation
        assert "metadata" in output
        assert output["metadata"]["aggregation_successful"] is False
        assert output["metadata"]["is_aggregated"] is True  # aggregation_result exists but failed

        # Check that templates_aggregation_metadata includes error info
        assert "templates_aggregation_metadata" in output
        assert "error" in output["templates_aggregation_metadata"]
        assert "metadata" in output["templates_aggregation_metadata"]

    def test_templates_aggregation_output_with_none_values(self):
        """Test templates aggregation output builders with None values."""
        from iris.orchestration.output_builders import build_simple_templates_aggregation_output

        # Create call_trace with None templates_aggregation
        call_trace = PipelineCallTraceStorage(results_names=["templates_aggregation"])
        call_trace.write_input([])  # Empty input
        call_trace.write("templates_aggregation", None)

        output = build_simple_templates_aggregation_output(call_trace)

        # Check that iris_template is None when templates_aggregation is None
        assert output["iris_template"] is None

        # Check that metadata is still generated
        assert "metadata" in output
        assert output["metadata"]["iris_version"] is not None
        assert output["metadata"]["input_templates_count"] == 0
        assert output["metadata"]["input_templates_image_ids"] == []

        # Check error field
        assert "error" in output
