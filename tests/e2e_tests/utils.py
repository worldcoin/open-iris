from typing import Any, Dict

import numpy as np


def compare_iris_pipeline_metadata_output(metadata_1: Dict[str, Any], metadata_2: Dict[str, Any]) -> None:
    """Compare two IRISPipeline outputs.

    Args:
        metadata_1 (Dict[str, Any]): pipeline's metadata output 1.
        metadata_2 (Dict[str, Any]): pipeline's metadata output 2.
    """
    assert metadata_2["image_size"] == metadata_1["image_size"]
    assert metadata_2["eye_side"] == metadata_1["eye_side"]

    np.testing.assert_almost_equal(
        metadata_2["eye_centers"]["pupil_center"],
        metadata_1["eye_centers"]["pupil_center"],
        decimal=4,
    )
    np.testing.assert_almost_equal(
        metadata_2["eye_centers"]["iris_center"],
        metadata_1["eye_centers"]["iris_center"],
        decimal=4,
    )
    np.testing.assert_almost_equal(
        list(metadata_2["pupil_to_iris_property"].values()),
        list(metadata_1["pupil_to_iris_property"].values()),
        decimal=4,
    )
    np.testing.assert_almost_equal(
        metadata_2["offgaze_score"],
        metadata_1["offgaze_score"],
        decimal=4,
    )
    np.testing.assert_almost_equal(
        metadata_2["eye_orientation"],
        metadata_1["eye_orientation"],
        decimal=4,
    )
    np.testing.assert_almost_equal(
        metadata_2["occlusion90"],
        metadata_1["occlusion90"],
        decimal=4,
    )
    np.testing.assert_almost_equal(
        metadata_2["occlusion30"],
        metadata_1["occlusion30"],
        decimal=4,
    )
    np.testing.assert_almost_equal(
        [
            metadata_2["iris_bbox"]["x_min"],
            metadata_2["iris_bbox"]["x_max"],
            metadata_2["iris_bbox"]["y_min"],
            metadata_2["iris_bbox"]["y_max"],
        ],
        [
            metadata_1["iris_bbox"]["x_min"],
            metadata_1["iris_bbox"]["x_max"],
            metadata_1["iris_bbox"]["y_min"],
            metadata_1["iris_bbox"]["y_max"],
        ],
        decimal=4,
    )


def compare_iris_pipeline_template_output(iris_template_1: Dict[str, Any], iris_template_2: Dict[str, Any]) -> None:
    """Compare two IRISPipeline template outputs.

    Args:
        iris_template_1 (Dict[str, Any]): pipeline's iris template output 1.
        iris_template_2 (Dict[str, Any]): pipeline's iris template output 2.
    """
    assert np.all(iris_template_2["iris_codes"] == iris_template_1["iris_codes"])
    assert np.all(iris_template_2["mask_codes"] == iris_template_1["mask_codes"])
    assert iris_template_2["iris_code_version"] == iris_template_1["iris_code_version"]


def compare_iris_pipeline_error_output(error_dict_1: Dict[str, str], error_dict_2: Dict[str, str]) -> None:
    """Compare two IRISPipeline error outputs.

    Args:
        error_dict_1 (Dict[str, str]): pipeline's error output 1.
        error_dict_2 (Dict[str, str]): pipeline's error output 2.
    """
    assert (error_dict_1 is None) == (error_dict_2 is None)
    if error_dict_1 is not None:
        assert error_dict_1["error_type"] == error_dict_2["error_type"]
        assert error_dict_1["traceback"] == error_dict_2["traceback"]
        assert error_dict_1["message"] == error_dict_2["message"]


def compare_simple_pipeline_template_output(iris_template_1: Dict[str, Any], iris_template_2: Dict[str, Any]) -> None:
    """Compare two IRISPipeline template outputs.

    Args:
        iris_template_1 (Dict[str, Any]): pipeline's iris template output 1.
        iris_template_2 (Dict[str, Any]): pipeline's iris template output 2.
    """
    assert np.all([ic1 == ic2 for ic1, ic2 in zip(iris_template_2.iris_codes, iris_template_1.iris_codes)])
    assert np.all([ic1 == ic2 for ic1, ic2 in zip(iris_template_2.mask_codes, iris_template_1.mask_codes)])
    assert iris_template_2.iris_code_version == iris_template_1.iris_code_version


def compare_simple_pipeline_outputs(pipeline_output_1: Dict[str, Any], pipeline_output_2: Dict[str, Any]):
    """Compare two IRISPipeline outputs for the Orb.

    Args:
        pipeline_output_1 (Dict[str, Any]): pipeline output 1.
        pipeline_output_2 (Dict[str, Any]): pipeline output 2.
    """
    compare_simple_pipeline_template_output(pipeline_output_1["iris_template"], pipeline_output_2["iris_template"])
    compare_iris_pipeline_metadata_output(pipeline_output_1["metadata"], pipeline_output_2["metadata"])
    compare_iris_pipeline_error_output(pipeline_output_1["error"], pipeline_output_2["error"])


def compare_iris_pipeline_outputs(pipeline_output_1: Dict[str, Any], pipeline_output_2: Dict[str, Any]):
    """Compare two IRISPipeline outputs for the Orb.

    Args:
        pipeline_output_1 (Dict[str, Any]): pipeline output 1.
        pipeline_output_2 (Dict[str, Any]): pipeline output 2.
    """
    compare_iris_pipeline_template_output(pipeline_output_1["iris_template"], pipeline_output_2["iris_template"])
    compare_iris_pipeline_metadata_output(pipeline_output_1["metadata"], pipeline_output_2["metadata"])
    compare_iris_pipeline_error_output(pipeline_output_1["error"], pipeline_output_2["error"])


def compare_debug_pipeline_outputs(pipeline_output_1: Dict[str, Any], pipeline_output_2: Dict[str, Any]):
    """Compare two IRISPipeline outputs for debugging.

    Args:
        pipeline_output_1 (Dict[str, Any]): pipeline output 1.
        pipeline_output_2 (Dict[str, Any]): pipeline output 2.
    """
    compare_simple_pipeline_template_output(pipeline_output_1["iris_template"], pipeline_output_2["iris_template"])
    compare_iris_pipeline_metadata_output(pipeline_output_1["metadata"], pipeline_output_2["metadata"])

    # Debug-specific intermediary outputs
    to_test = {
        "normalized_iris": ["normalized_image", "normalized_mask"],
        "iris_response": ["iris_responses", "mask_responses"],
        "extrapolated_polygons": ["pupil", "iris", "eyeball"],
    }
    for key, values in to_test.items():
        for value in values:
            np.testing.assert_almost_equal(
                pipeline_output_1[key][value],
                pipeline_output_2[key][value],
                decimal=4,
            )
    np.testing.assert_almost_equal(
        pipeline_output_1["segmentation_map"]["predictions"],
        pipeline_output_2["segmentation_map"]["predictions"],
        decimal=4,
    )
