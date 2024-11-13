import traceback
from typing import Any, Dict, Optional

from iris._version import __version__
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import ImmutableModel


def build_simple_orb_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Build the output for the Orb.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call results storage.

    Returns:
        Dict[str, Any]: {
        "iris_template": (Optional[IrisTemplate]) the iris template object if the pipeline succeeded,
        "error": (Optional[Dict]) the error dict if the pipeline returned an error,
        "metadata": (Dict) the metadata dict,
        }.
    """
    metadata = __get_metadata(call_trace=call_trace)
    error = __get_error(call_trace=call_trace)
    iris_template = call_trace["encoder"]

    output = {
        "error": error,
        "iris_template": iris_template,
        "metadata": metadata,
    }

    return output


def build_orb_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Build the output for the Orb.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call results storage.

    Returns:
        Dict[str, Any]: {
        "iris_template": (Optional[Dict]) the iris template dict if the pipeline succeeded.
        "error": (Optional[Dict]) the error dict if the pipeline returned an error.
        "metadata": (Dict) the metadata dict.
        }.
    """
    output = build_simple_orb_output(call_trace)
    output["iris_template"] = __safe_serialize(output["iris_template"])

    return output


def build_simple_debugging_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Build the simplest output for debugging purposes.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call results storage.

    Returns:
        Dict[str, Any]: Returns data to be stored in MongoDB.
    """
    iris_template = call_trace["encoder"]

    metadata = __get_metadata(call_trace=call_trace)
    error = __get_error(call_trace=call_trace)

    segmap = call_trace["segmentation"]
    geometry_mask, noise_mask = (None, None) if call_trace["segmentation_binarization"] is None else call_trace["segmentation_binarization"]
    extrapolated_polygons = call_trace["geometry_estimation"]
    normalized_iris = call_trace["normalization"]
    iris_response = call_trace["filter_bank"]

    return {
        "iris_template": iris_template,
        "metadata": metadata,
        "segmentation_map": __safe_serialize(segmap),
        "segmentation_binarization": {
            "geometry": __safe_serialize(geometry_mask),
            "noise": __safe_serialize(noise_mask),
        },
        "extrapolated_polygons": __safe_serialize(extrapolated_polygons),
        "normalized_iris": __safe_serialize(normalized_iris),
        "iris_response": __safe_serialize(iris_response),
        "error": error,
    }


def __safe_serialize(object: Optional[ImmutableModel]) -> Optional[Dict[str, Any]]:
    """Serialize an object.

    Args:
        object (Optional[ImmutableModel]): Object to be serialized.

    Raises:
        NotImplementedError: Raised if object is not serializable.

    Returns:
        Optional[Dict[str, Any]]: Serialized object.
    """
    if object is None:
        return None
    elif isinstance(object, ImmutableModel):
        return object.serialize()
    elif isinstance(object, (list, tuple)):
        return [__safe_serialize(sub_object) for sub_object in object]
    else:
        raise NotImplementedError(f"Object of type {type(object)} is not serializable.")


def __get_metadata(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
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
        "eye_centers": __safe_serialize(call_trace["eye_center_estimation"]),
        "pupil_to_iris_property": __safe_serialize(call_trace["pupil_to_iris_property_estimation"]),
        "offgaze_score": __safe_serialize(call_trace["offgaze_estimation"]),
        "eye_orientation": __safe_serialize(call_trace["eye_orientation"]),
        "occlusion90": __safe_serialize(call_trace["occlusion90_calculator"]),
        "occlusion30": __safe_serialize(call_trace["occlusion30_calculator"]),
        "iris_bbox": __safe_serialize(call_trace["bounding_box_estimation"]),
    }


def build_debugging_output(call_trace: PipelineCallTraceStorage) -> Dict[str, Any]:
    """Build the output for debugging purposes.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call results storage.

    Returns:
        Dict[str, Any]: Returns data to be stored in MongoDB.
    """
    output = build_simple_debugging_output(call_trace)

    serialized_iris_template = __safe_serialize(output["iris_template"])
    output["iris_template"] = serialized_iris_template

    return output


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
