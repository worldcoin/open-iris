import traceback
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import numpy as np

from iris._version import __version__
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import ImmutableModel, OutputFieldSpec


def _nested_safe_serialize(obj: Any) -> Any:
    """
    Apply __safe_serialize to obj, handling nested dicts by recursing into values.
    Lists and tuples are handled by __safe_serialize itself.
    """
    if obj is None:
        return None
    # Handle mappings by serializing each value
    if isinstance(obj, Mapping):
        return {k: _nested_safe_serialize(v) for k, v in obj.items()}
    # Fallback to the existing helper (handles ImmutableModel, list, tuple)
    return __safe_serialize(obj)


def _build_from_spec(call_trace: PipelineCallTraceStorage, spec: List[OutputFieldSpec]) -> Dict[str, Any]:
    """
    Generic builder that constructs an output dict based on a list of OutputFieldSpec.

    Args:
        call_trace (PipelineCallTraceStorage): The pipeline call trace storage object.
        spec (List[OutputFieldSpec]): A list of OutputFieldSpec defining how to extract and optionally serialize each field.

    Returns:
        Dict[str, Any]: A dict mapping each spec.key to the (optionally serialized) extracted value.
    """
    output: Dict[str, Any] = {}
    for field in spec:
        # Extract the raw value using the provided extractor function
        val = field.extractor(call_trace)
        # If requested, wrap complex objects in a safe-serialize step
        if field.safe_serialize:
            val = _nested_safe_serialize(val)
        output[field.key] = val
    return output


def __safe_serialize(object: Optional[Any]) -> Optional[Any]:
    """Serialize an object.

    Args:
        object (Optional[Any]): Object to be serialized.

    Raises:
        NotImplementedError: Raised if object is not serializable.

    Returns:
        Optional[Any]: Serialized object.
    """
    if object is None:
        return None
    elif isinstance(object, ImmutableModel):
        return object.serialize()
    elif isinstance(object, (list, tuple)):
        return [__safe_serialize(sub_object) for sub_object in object]
    elif isinstance(object, np.ndarray):
        return object.tolist()
    else:
        raise NotImplementedError(f"Object of type {type(object)} is not serializable.")


def __get_iris_pipeline_metadata(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Produce metadata output from a call_trace.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call trace.

    Returns:
        Dict[str, Any]: Metadata dictionary.
    """
    ir_image = call_trace.get_input()

    return {
        "iris_version": __version__,
        "image_size": (ir_image.width, ir_image.height),
        "eye_side": ir_image.eye_side,
        "eye_centers": __safe_serialize(call_trace.get("eye_center_estimation")),
        "pupil_to_iris_property": __safe_serialize(call_trace.get("pupil_to_iris_property_estimation")),
        "offgaze_score": __safe_serialize(call_trace.get("offgaze_estimation")),
        "eye_orientation": __safe_serialize(call_trace.get("eye_orientation")),
        "occlusion90": __safe_serialize(call_trace.get("occlusion90_calculator")),
        "occlusion30": __safe_serialize(call_trace.get("occlusion30_calculator")),
        "iris_bbox": __safe_serialize(call_trace.get("bounding_box_estimation")),
        "sharpness_score": __safe_serialize(call_trace.get("sharpness_estimation")),
    }


def __get_error(call_trace: PipelineCallTraceStorage) -> Optional[Dict[str, Any]]:
    """Produce error output from a call_trace.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call trace.

    Returns:
        Optional[Dict[str, Any]]: Optional error dictionary if such occured.
    """
    exception = call_trace.get_error()
    error = None

    if isinstance(exception, Exception):
        error = {
            "error_type": type(exception).__name__,
            "message": str(exception),
            "traceback": "".join(traceback.format_tb(exception.__traceback__)),
        }

    return error


def __get_multiframe_aggregation_metadata(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Produce multiframe aggregation metadata output from a call_trace.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call trace.

    Returns:
        Dict[str, Any]: Metadata dictionary.
    """
    templates = call_trace.get_input()
    aligned_templates = call_trace.get("templates_alignment")
    identity_filtered_templates = call_trace.get("identity_validation")

    return {
        "iris_version": __version__,
        "input_templates_count": len(templates) if templates is not None else None,
        "aligned_templates": {
            "reference_template_id": aligned_templates.reference_template_id if aligned_templates is not None else None,
            "distances": __safe_serialize(aligned_templates.distances) if aligned_templates is not None else None,
        },
        "post_identity_filter_templates_count": len(identity_filtered_templates)
        if identity_filtered_templates is not None
        else None,
    }


# =============================================================================
# Specs for different output variants
# =============================================================================

# Simple ORB output: raw iris_template, error info, and metadata
IRIS_PIPE_SIMPLE_ORB_OUTPUT_SPEC = [
    OutputFieldSpec(key="error", extractor=__get_error, safe_serialize=False),
    OutputFieldSpec(key="iris_template", extractor=lambda ct: ct.get("encoder"), safe_serialize=False),
    OutputFieldSpec(key="metadata", extractor=__get_iris_pipeline_metadata, safe_serialize=False),
]

IRIS_PIPE_ORB_OUTPUT_SPEC = [
    OutputFieldSpec(key="error", extractor=__get_error, safe_serialize=False),
    OutputFieldSpec(key="iris_template", extractor=lambda ct: ct.get("encoder"), safe_serialize=True),
    OutputFieldSpec(key="metadata", extractor=__get_iris_pipeline_metadata, safe_serialize=False),
]

# Debugging output: includes various intermediate pipeline results
IRIS_PIPE_DEBUG_OUTPUT_SPEC = [
    OutputFieldSpec(key="iris_template", extractor=lambda ct: ct.get("encoder"), safe_serialize=False),
    OutputFieldSpec(key="metadata", extractor=__get_iris_pipeline_metadata, safe_serialize=False),
    OutputFieldSpec(key="segmentation_map", extractor=lambda ct: ct.get("segmentation"), safe_serialize=True),
    OutputFieldSpec(
        key="segmentation_binarization",
        extractor=lambda ct: {
            "geometry": None if ct.get("segmentation_binarization") is None else ct.get("segmentation_binarization")[0],
            "noise": None if ct.get("segmentation_binarization") is None else ct.get("segmentation_binarization")[1],
        },
        safe_serialize=True,
    ),
    OutputFieldSpec(
        key="extrapolated_polygons", extractor=lambda ct: ct.get("geometry_estimation"), safe_serialize=True
    ),
    OutputFieldSpec(key="normalized_iris", extractor=lambda ct: ct.get("normalization"), safe_serialize=True),
    OutputFieldSpec(key="iris_response", extractor=lambda ct: ct.get("filter_bank"), safe_serialize=True),
    OutputFieldSpec(
        key="iris_response_refined", extractor=lambda ct: ct.get("iris_response_refinement"), safe_serialize=True
    ),
    OutputFieldSpec(key="error", extractor=__get_error, safe_serialize=False),
]

MULTIFRAME_AGG_ORB_OUTPUT_SPEC = [
    OutputFieldSpec(key="error", extractor=__get_error, safe_serialize=False),
    OutputFieldSpec(
        key="iris_template",
        extractor=lambda ct: ct.get("templates_aggregation").as_iris_template()
        if ct.get("templates_aggregation") is not None
        else None,
        safe_serialize=True,
    ),
    OutputFieldSpec(key="metadata", extractor=__get_multiframe_aggregation_metadata, safe_serialize=False),
]

MULTIFRAME_AGG_SIMPLE_ORB_OUTPUT_SPEC = [
    OutputFieldSpec(key="error", extractor=__get_error, safe_serialize=False),
    OutputFieldSpec(
        key="iris_template",
        extractor=lambda ct: ct.get("templates_aggregation").as_iris_template()
        if ct.get("templates_aggregation") is not None
        else None,
        safe_serialize=False,
    ),
    OutputFieldSpec(key="metadata", extractor=__get_multiframe_aggregation_metadata, safe_serialize=False),
]

# =============================================================================
# Builder functions leveraging the generic engine
# =============================================================================


def build_simple_iris_pipeline_orb_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Construct simple ORB output: raw iris_template, error, and metadata."""
    return _build_from_spec(call_trace, IRIS_PIPE_SIMPLE_ORB_OUTPUT_SPEC)


def build_iris_pipeline_orb_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Construct ORB output with serialized iris_template."""
    return _build_from_spec(call_trace, IRIS_PIPE_ORB_OUTPUT_SPEC)


def build_simple_iris_pipeline_debugging_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Construct debugging output with intermediate results (raw values)."""
    return _build_from_spec(call_trace, IRIS_PIPE_DEBUG_OUTPUT_SPEC)


def build_iris_pipeline_debugging_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """
    Construct full debugging output: wrap simple_debugging and ensure
    the iris_template is also safely serialized.
    """
    output = build_simple_iris_pipeline_debugging_output(call_trace)
    output["iris_template"] = __safe_serialize(output.get("iris_template"))
    return output


def build_aggregation_multiframe_orb_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Construct multiframe aggregation ORB output with safe serialization."""
    return _build_from_spec(call_trace, MULTIFRAME_AGG_ORB_OUTPUT_SPEC)


def build_simple_multiframe_aggregation_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Construct simple multiframe aggregation output (raw values)."""
    return _build_from_spec(call_trace, MULTIFRAME_AGG_SIMPLE_ORB_OUTPUT_SPEC)
