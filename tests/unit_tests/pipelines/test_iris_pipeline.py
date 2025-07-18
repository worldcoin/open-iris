import os
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import onnxruntime as ort
import pytest
import yaml
from _pytest.fixtures import FixtureRequest

import iris
from iris import __version__
from iris.callbacks.pipeline_trace import NodeResultsWriter, PipelineCallTraceStorage
from iris.io.class_configs import Algorithm
from iris.io.errors import IRISPipelineError
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import raise_error_manager, store_error_manager
from iris.orchestration.output_builders import (
    build_iris_pipeline_orb_output,
    build_simple_iris_pipeline_debugging_output,
)
from iris.orchestration.pipeline_dataclasses import PipelineClass
from iris.pipelines.iris_pipeline import IRISPipeline
from iris.utils.base64_encoding import base64_encode_str
from tests.e2e_tests.utils import (
    compare_debug_pipeline_outputs,
    compare_iris_pipeline_outputs,
    compare_simple_pipeline_outputs,
)


@pytest.fixture
def ir_image() -> np.ndarray:
    ir_image_path = os.path.join(os.path.dirname(__file__), "mocks", "anonymized.png")
    img_data = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)

    return img_data


@pytest.fixture
def fake_ir_image() -> np.ndarray:
    img_data = np.random.random((1080, 1440))

    return img_data


def load_config_str(config_name) -> str:
    """Load a yaml config file"""
    config_path = os.path.join(os.path.dirname(__file__), "mocks", config_name)

    with open(config_path, "r") as f:
        config_str = f.read()

    return config_str


@pytest.mark.parametrize(
    "config",
    [
        (
            {
                "metadata": {
                    "pipeline_name": "iris_pipeline",
                    "iris_version": "14.28.57",
                },
                "pipeline": [],
            }
        ),
        (
            {
                "metadata": {
                    "pipeline_name": "iris_pipeline",
                    "iris_version": __version__,
                },
                "pipeline": [
                    {
                        "name": "segmentation",
                        "algorithm": {
                            "class_name": "iris.nodes.segmentation.onnx_multilabel_segmentation.ONNXMultilabelSegmentation",
                            "params": {},
                        },
                        "inputs": [{"name": "image", "source_node": "input"}],
                        "callbacks": None,
                        "seed": None,
                    },
                    {
                        "name": "segmentation",
                        "algorithm": {
                            "class_name": "iris.nodes.segmentation.onnx_multilabel_segmentation.ONNXMultilabelSegmentation",
                            "params": {},
                        },
                        "inputs": [{"name": "image", "source_node": "input"}],
                        "callbacks": None,
                        "seed": None,
                    },
                    {
                        "name": "binarization",
                        "algorithm": {
                            "class_name": "iris.nodes.binarization.specular_reflection_detection.SpecularReflectionDetection",
                            "params": {},
                        },
                        "inputs": [{"name": "ir_image", "source_node": "input"}],
                        "callbacks": None,
                        "seed": None,
                    },
                ],
            }
        ),
    ],
    ids=["wrong version", "duplicate node name"],
)
def test_pipeline_sanity_check_fails(config: Dict[str, Any]):
    with pytest.raises(IRISPipelineError):
        _ = IRISPipeline(config=config)


@pytest.mark.parametrize(
    "config,expectation",
    [
        (
            None,
            does_not_raise(),
        ),
        (
            "",
            does_not_raise(),
        ),
        (
            "test:\n  - 1\n  - 2",
            does_not_raise(),
        ),
        (
            f"metadata:\n  pipeline_name: iris_pipeline\n  iris_version: {__version__}\n\npipeline: []",
            does_not_raise(),
        ),
        (
            "test:\n  - a: 1\n - b: ",
            pytest.raises(ValueError),
        ),
        (
            {"not": ["a", "str"]},
            pytest.raises(ValueError),
        ),
    ],
    ids=[
        "None config",
        "Empty str config",
        "properly-formated config 1",
        "properly-formated config 2",
        "poorly-formated config",
        "not a string",
    ],
)
def test_load_config(config: Optional[str], expectation):
    with expectation:
        cfg = IRISPipeline.load_config(config)
        assert isinstance(cfg, dict)


@pytest.mark.parametrize(
    "config,expectation",
    [
        (
            None,
            does_not_raise(),
        ),
        (
            f"metadata:\n  pipeline_name: iris_pipeline\n  iris_version: {__version__}\n\npipeline: []",
            does_not_raise(),
        ),
        (
            load_config_str("incoherent_pipeline_1.yml"),
            pytest.raises(IRISPipelineError),
        ),
        (
            load_config_str("incoherent_pipeline_2.yml"),
            pytest.raises(IRISPipelineError),
        ),
        (
            load_config_str("incoherent_pipeline_3.yml"),
            pytest.raises(IRISPipelineError),
        ),
    ],
    ids=[
        "None config = default",
        "Empty config",
        "One node has an input non-declared",
        "One node has a nested input non-declared",
        "Regular pipeline with two nodes inverted",
    ],
)
def test_check_pipeline_coherency_fails(config: Optional[str], expectation):
    with expectation:
        _ = IRISPipeline(config=config)


@pytest.mark.parametrize(
    "input,env,expectation",
    [
        (
            "fake_ir_image",
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            pytest.raises(Exception),
        ),
        (
            "ir_image",
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            does_not_raise(),
        ),
        (
            "fake_ir_image",
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            does_not_raise(),
        ),
        (
            "ir_image",
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            does_not_raise(),
        ),
    ],
    ids=[
        "pipeline fails w/ raise_error_manager",
        "pipeline runs w/ raise_error_manager",
        "pipeline fails w/ store_error_manager",
        "pipeline runs w/ store_error_manager",
    ],
)
def test_error_manager(input: str, env: Environment, expectation, request: FixtureRequest):
    ir_image = request.getfixturevalue(input)
    iris_pipeline = IRISPipeline(env=env)

    with expectation:
        _ = iris_pipeline(ir_image, eye_side="left")


@pytest.mark.parametrize(
    "input,env,expected_non_null_call_trace,expected_non_null_outputs",
    [
        (
            "ir_image",
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            set(IRISPipeline().call_trace._storage.keys())
            - set(["eye_centers_inside_image_validator", "extrapolated_polygons_inside_image_validator", "error"]),
            ["iris_template", "metadata", "normalized_image"],
        ),
        (
            "fake_ir_image",
            Environment(
                pipeline_output_builder=build_simple_iris_pipeline_debugging_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            ["input", "segmentation", "segmentation_binarization", "error"],
            ["segmentation_map", "segmentation_binarization", "error", "metadata"],
        ),
        (
            "ir_image",
            Environment(
                pipeline_output_builder=build_simple_iris_pipeline_debugging_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            set(IRISPipeline().call_trace._storage.keys())
            - set(["eye_centers_inside_image_validator", "extrapolated_polygons_inside_image_validator", "error"]),
            [
                "segmentation_map",
                "segmentation_binarization",
                "extrapolated_polygons",
                "normalized_iris",
                "iris_response",
                "iris_response_refined",
                "landmarks",
                "iris_template",
                "status",
                "metadata",
            ],
        ),
    ],
    ids=[
        "pipeline runs w/ raise_error_manager",
        "pipeline fails w/ store_error_manager",
        "pipeline runs w/ store_error_manager",
    ],
)
def test_interrupted_pipeline_has_the_right_call_trace(
    input: str,
    env: Environment,
    expected_non_null_call_trace: List[str],
    expected_non_null_outputs: List[str],
    request: FixtureRequest,
):
    ir_image = request.getfixturevalue(input)
    iris_pipeline = IRISPipeline(env=env)
    result = iris_pipeline(ir_image, eye_side="left")

    # Test for None / non-None values in call_trace
    for node_name, intermediary_result in iris_pipeline.call_trace._storage.items():
        if intermediary_result is None:
            assert node_name not in expected_non_null_call_trace
        else:
            assert node_name in expected_non_null_call_trace

    # Test for None / non-None values in final output
    for key, value in result.items():
        if value is None:
            assert key not in expected_non_null_outputs
        else:
            assert key in expected_non_null_outputs


def test_init_pipeline_tracing() -> None:
    iris_pipeline = IRISPipeline()

    alg_names: List[str] = [node.name for node in iris_pipeline.params.pipeline]
    alg_objs: List[Algorithm] = list(iris_pipeline.nodes.values())

    expected_keys_in_storage = [
        *alg_names,
        PipelineCallTraceStorage.INPUT_KEY_NAME,
        PipelineCallTraceStorage.ERROR_KEY_NAME,
    ]

    assert sorted(expected_keys_in_storage) == sorted(iris_pipeline.call_trace._storage.keys())

    for alg_obj in alg_objs:
        for cb in alg_obj._callbacks:
            if isinstance(cb, NodeResultsWriter):
                has_writer = True
                break

        if not has_writer:
            assert False, "IRISPipeline node not instantiate with NodeResultsWriter."


def test_call_trace_clearance(ir_image: np.ndarray) -> None:
    iris_pipeline = IRISPipeline()

    first_call = iris_pipeline(ir_image, eye_side="left")
    second_call = iris_pipeline(ir_image, eye_side="right")

    assert first_call["metadata"]["eye_side"] != second_call["metadata"]["eye_side"]


@pytest.mark.parametrize(
    "node_class,algorithm_params,callbacks,expected_callbacks_params_values,env",
    [
        (
            "iris.nodes.geometry_refinement.smoothing.Smoothing",
            {},
            None,
            [],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                disabled_qa=[],
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
        (
            "iris.nodes.geometry_refinement.smoothing.Smoothing",
            {"dphi": 14.2, "kernel_size": 8.57},
            None,
            [],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                disabled_qa=[iris.nodes.validators.object_validators.IsPupilInsideIrisValidator],
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
        (
            "iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation",
            {"dphi": 142.857},
            [
                PipelineClass(
                    class_name="iris.nodes.validators.object_validators.PolygonsLengthValidator",
                    params={"min_iris_length": 300, "min_pupil_length": 300},
                ),
                PipelineClass(
                    class_name="iris.nodes.validators.object_validators.PolygonsLengthValidator",
                    params={"min_iris_length": 400, "min_pupil_length": 400},
                ),
            ],
            [{"min_iris_length": 300, "min_pupil_length": 300}, {"min_iris_length": 400, "min_pupil_length": 400}],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                disabled_qa=[iris.nodes.validators.object_validators.IsPupilInsideIrisValidator],
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
        (
            "iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation",
            {"dphi": 142.857},
            None,
            [],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                disabled_qa=[iris.nodes.validators.object_validators.PolygonsLengthValidator],
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
        (
            "iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation",
            {"dphi": 1.5},
            [
                PipelineClass(
                    class_name="iris.nodes.validators.object_validators.PolygonsLengthValidator",
                    params={"min_iris_length": 300, "min_pupil_length": 300},
                ),
                PipelineClass(
                    class_name="iris.nodes.validators.object_validators.OcclusionValidator",
                    params={"min_allowed_occlusion": 0.5},
                ),
            ],
            [{"min_iris_length": 300, "min_pupil_length": 300}],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                disabled_qa=[
                    iris.nodes.validators.object_validators.OcclusionValidator,
                    iris.nodes.validators.cross_object_validators.EyeCentersInsideImageValidator,
                ],
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
        (
            "iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation",
            {"dphi": 1.5},
            [
                PipelineClass(
                    class_name="iris.nodes.validators.object_validators.PolygonsLengthValidator",
                    params={"min_iris_length": 300, "min_pupil_length": 300},
                ),
                PipelineClass(
                    class_name="iris.nodes.validators.object_validators.OcclusionValidator",
                    params={"min_allowed_occlusion": 0.5},
                ),
            ],
            [],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                disabled_qa=[
                    iris.nodes.validators.object_validators.OcclusionValidator,
                    iris.nodes.validators.object_validators.PolygonsLengthValidator,
                    iris.nodes.validators.cross_object_validators.EyeCentersInsideImageValidator,
                ],
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
    ],
    ids=[
        "Node without params without callbacks",
        "Node with params without callbacks",
        "Node with params with several callbacks",
        "Node with params without callbacks + callbacks disabled",
        "Node with params with several callbacks + some callbacks disabled",
        "Node with params with several callbacks + all callbacks disabled",
    ],
)
def test_instanciate_node(
    node_class: str,
    algorithm_params: Dict[str, Any],
    callbacks: Optional[List[PipelineClass]],
    expected_callbacks_params_values: List[Dict[str, Any]],
    env: Environment,
) -> None:
    """Tests IRISPipeline's instanciate_node function in various scenarios.

    The goals of this tests are
      * to ensure that nodes are created as expected,
      * to ensure that callbacks get filtered out correctly through the Environment.disable_qa mechanism
    """
    config = f"metadata:\n  pipeline_name: iris_pipeline\n  iris_version: {__version__}\n\npipeline: []"
    iris_pipeline = IRISPipeline(config=config, env=env)

    node = iris_pipeline._instanciate_node(
        node_class=node_class, algorithm_params=algorithm_params, callbacks=callbacks
    )

    # Check if the created node has the right type
    assert isinstance(node, eval(node_class))

    # Check if the created node has the right parameters
    for param_name, param_value in algorithm_params.items():
        assert getattr(node.params, param_name) == param_value

    # Check if all created callbacks have the right parameters
    assert len(node._callbacks) == len(expected_callbacks_params_values)

    for result_cb, expected_cb in zip(node._callbacks, expected_callbacks_params_values):
        for param_name, param_value in expected_cb.items():
            assert getattr(result_cb.params, param_name) == param_value


@pytest.mark.parametrize(
    "pipeline,expected_built_pipeline,env",
    [
        (
            [],
            [],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
        (
            [
                {
                    "name": "vectorization",
                    "algorithm": {
                        "class_name": "iris.nodes.vectorization.contouring.ContouringAlgorithm",
                        "params": {},
                    },
                    "inputs": [{"name": "image", "source_node": "input"}],
                    "callbacks": None,
                },
                {
                    "name": "geometry_estimation",
                    "algorithm": {
                        "class_name": "iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation",
                        "params": {"dphi": 142.857},
                    },
                    "inputs": [{"name": "image", "source_node": "input"}],
                    "callbacks": [
                        {
                            "class_name": "iris.nodes.validators.object_validators.PolygonsLengthValidator",
                            "params": {"min_iris_length": 142, "min_pupil_length": 857},
                        }
                    ],
                },
            ],
            [
                iris.nodes.vectorization.contouring.ContouringAlgorithm(),
                iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation(
                    dphi=142.857,
                    callbacks=[
                        iris.nodes.validators.object_validators.PolygonsLengthValidator(
                            min_iris_length=142, min_pupil_length=857
                        )
                    ],
                ),
            ],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
        (
            [
                {
                    "name": "vectorization",
                    "algorithm": {
                        "class_name": "iris.nodes.vectorization.contouring.ContouringAlgorithm",
                        "params": {},
                    },
                    "inputs": [{"name": "image", "source_node": "input"}],
                    "callbacks": None,
                },
                {
                    "name": "geometry_estimation",
                    "algorithm": {
                        "class_name": "iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation",
                        "params": {"dphi": 142.857},
                    },
                    "inputs": [{"name": "image", "source_node": "input"}],
                    "callbacks": [
                        {
                            "class_name": "iris.nodes.validators.object_validators.PolygonsLengthValidator",
                            "params": {"min_iris_length": 142, "min_pupil_length": 857},
                        }
                    ],
                },
            ],
            [
                iris.nodes.vectorization.contouring.ContouringAlgorithm(),
                iris.nodes.geometry_estimation.linear_extrapolation.LinearExtrapolation(dphi=142.857, callbacks=[]),
            ],
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=raise_error_manager,
                disabled_qa=[
                    iris.nodes.validators.object_validators.OffgazeValidator,
                    iris.nodes.validators.object_validators.SharpnessValidator,
                    iris.nodes.validators.object_validators.PolygonsLengthValidator,
                ],
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        ),
    ],
    ids=[
        "empty pipeline",
        "several nodes, unfiltered",
        "several nodes, some cross_object_validators filtered + some node's callbacks filtered too",
    ],
)
def test_instanciate_nodes(
    pipeline: List[Dict[str, Any]], expected_built_pipeline: List[Algorithm], env: Environment
) -> None:
    """Tests IRISPipeline's instanciate_nodes function in various scenarios.

    The goals of this tests are
      * to ensure that the list of nodes is created as expected,
      * to ensure that callbacks get filtered out correctly through the Environment.disable_qa mechanism
      * to ensure that nodes get filtered out correctly through the Environment.disable_qa mechanism
    """
    b = "\n"
    config = f"metadata:\n  pipeline_name: iris_pipeline\n  iris_version: {__version__}\npipeline:{b if len(pipeline) > 0 else ' '}{yaml.dump(pipeline)}"

    iris_pipeline = IRISPipeline(config=config, env=env)

    nodes = iris_pipeline._instanciate_nodes()

    for computed_node, expected_node in zip(nodes.values(), expected_built_pipeline):
        assert isinstance(computed_node, type(expected_node))
        assert computed_node.params == expected_node.params
        assert len(computed_node._callbacks) == len(expected_node._callbacks)

        for computed_cb, expected_cb in zip(computed_node._callbacks, expected_node._callbacks):
            assert isinstance(computed_cb, type(expected_cb))
            assert computed_cb.params == expected_cb.params


@pytest.mark.parametrize(
    "config,expected_pipeline_name",
    [
        (
            base64_encode_str(
                f"metadata:\n  pipeline_name: v1.5.1_pipeline\n  iris_version: {__version__}\n\npipeline: []"
            ),
            "v1.5.1_pipeline",
        ),
        (None, None),
    ],
    ids=["specified pipeline", "default pipeline"],
)
def test_load_from_config(config: Dict[str, str], expected_pipeline_name: str) -> None:
    res = IRISPipeline.load_from_config(config=config)

    if expected_pipeline_name is not None:
        assert res["agent"].params.metadata.pipeline_name == expected_pipeline_name
    else:
        assert res["agent"] is None


@pytest.mark.parametrize(
    "env,comparison_func",
    [
        (
            Environment(
                pipeline_output_builder=build_iris_pipeline_orb_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            compare_iris_pipeline_outputs,
        ),
        (
            Environment(
                pipeline_output_builder=build_simple_iris_pipeline_debugging_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            compare_debug_pipeline_outputs,
        ),
    ],
    ids=["orb_output_builder", "debugging_output_builder"],
)
def test_estimate_and_run_produce_same_output(ir_image: np.ndarray, env: Environment, comparison_func):
    """Test that estimate() and run() methods produce identical outputs for IRISPipeline."""
    # Create pipeline instance
    iris_pipeline = IRISPipeline(env=env)

    # Test with left eye
    estimate_result_left = iris_pipeline.estimate(ir_image, eye_side="left")
    run_result_left = iris_pipeline.run(ir_image, eye_side="left")

    # Assert that both methods produce the same output for left eye
    comparison_func(estimate_result_left, run_result_left)

    # Test with right eye
    estimate_result_right = iris_pipeline.estimate(ir_image, eye_side="right")
    run_result_right = iris_pipeline.run(ir_image, eye_side="right")

    # Assert that both methods produce the same output for right eye
    comparison_func(estimate_result_right, run_result_right)


def test_estimate_and_run_with_additional_args(ir_image: np.ndarray):
    """Test that estimate() and run() produce same output with additional args and kwargs."""
    # Create pipeline instance with debugging environment for more predictable output
    env = Environment(
        pipeline_output_builder=build_simple_iris_pipeline_debugging_output,
        error_manager=store_error_manager,
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )
    iris_pipeline = IRISPipeline(env=env)

    # Test with additional arguments (though IRISPipeline doesn't use them, they should be passed through)
    extra_arg = "extra_positional"
    extra_kwarg = {"extra": "keyword"}

    estimate_result = iris_pipeline.estimate(ir_image, "left", extra_arg, extra_param=extra_kwarg)
    run_result = iris_pipeline.run(ir_image, "left", extra_arg, extra_param=extra_kwarg)

    # Assert that both methods produce the same output
    compare_debug_pipeline_outputs(estimate_result, run_result)


def test_estimate_calls_run_internally(ir_image: np.ndarray):
    """Test that estimate() method internally calls run() method."""
    # Create pipeline instance
    iris_pipeline = IRISPipeline()

    # Mock the run method to verify it's called by estimate
    original_run = iris_pipeline.run
    iris_pipeline.run = Mock(return_value={"mocked": "result"})

    # Call estimate
    result = iris_pipeline.estimate(ir_image, "left", "arg1", kwarg1="value1")

    # Verify run was called with the same arguments
    iris_pipeline.run.assert_called_once_with(ir_image, "left", "arg1", kwarg1="value1")
    assert result == {"mocked": "result"}

    # Restore original run method
    iris_pipeline.run = original_run


def test_estimate_run_equivalence_with_call_method(ir_image: np.ndarray):
    """Test that estimate(), run(), and __call__() all produce the same output."""
    # Create pipeline instance with orb output builder (default)
    iris_pipeline = IRISPipeline()

    # Get results from all three methods
    estimate_result = iris_pipeline.estimate(ir_image, eye_side="left")
    run_result = iris_pipeline.run(ir_image, eye_side="left")
    call_result = iris_pipeline(ir_image, eye_side="left")

    # All three should produce identical results
    compare_simple_pipeline_outputs(estimate_result, run_result)
    compare_simple_pipeline_outputs(run_result, call_result)
    compare_simple_pipeline_outputs(estimate_result, call_result)


def test_iris_pipeline_has_correct_package_version():
    """Test that IRISPipeline has the correct PACKAGE_VERSION."""
    assert hasattr(IRISPipeline, "PACKAGE_VERSION")
    assert IRISPipeline.PACKAGE_VERSION == __version__
    assert isinstance(IRISPipeline.PACKAGE_VERSION, str)
    assert len(IRISPipeline.PACKAGE_VERSION) > 0


def test_correct_version_passes_validation():
    """Test that correct version passes validation during pipeline creation."""
    config = {
        "metadata": {
            "pipeline_name": "test_pipeline",
            "iris_version": __version__,  # Correct version
        },
        "pipeline": [],
    }

    # Should not raise any exception
    pipeline = IRISPipeline(config=config)
    assert pipeline.params.metadata.iris_version == __version__


def test_wrong_version_raises_error():
    """Test that wrong version raises error during pipeline creation."""
    config = {
        "metadata": {
            "pipeline_name": "test_pipeline",
            "iris_version": "999.0.0",  # Wrong version
        },
        "pipeline": [],
    }

    with pytest.raises(IRISPipelineError) as exc_info:
        IRISPipeline(config=config)

    error_message = str(exc_info.value)
    assert "Wrong config version" in error_message
    assert __version__ in error_message
    assert "999.0.0" in error_message


def test_version_validator_returns_metadata_on_success():
    """Test that version validator returns metadata object when validation passes."""
    # Create a mock metadata object
    mock_metadata = Mock()
    mock_metadata.iris_version = __version__

    # The validator should return the metadata object unchanged
    result = IRISPipeline.Parameters._version_check(mock_metadata, {})
    assert result == mock_metadata


def test_custom_pipeline_with_different_version():
    """Test that custom pipeline with different PACKAGE_VERSION is created correctly."""

    class CustomPipeline(IRISPipeline):
        PACKAGE_VERSION = "2.0.0"

    # Verify that the custom pipeline has its own version
    assert CustomPipeline.PACKAGE_VERSION == "2.0.0"
    assert CustomPipeline.PACKAGE_VERSION != IRISPipeline.PACKAGE_VERSION


def test_custom_pipeline_without_package_version_uses_parent():
    """Test that custom pipeline without PACKAGE_VERSION uses parent's version."""

    class CustomPipelineWithoutVersion(IRISPipeline):
        # Deliberately not setting PACKAGE_VERSION
        pass

    # Should inherit from parent IRISPipeline
    assert CustomPipelineWithoutVersion.PACKAGE_VERSION == IRISPipeline.PACKAGE_VERSION
    assert CustomPipelineWithoutVersion.PACKAGE_VERSION == __version__


@pytest.mark.parametrize(
    "config,expectation",
    [
        (
            base64_encode_str(
                f"metadata:\n  pipeline_name: v1.5.1_pipeline\n  iris_version: {__version__}\n\npipeline: []"
            ),
            does_not_raise(),
        ),
        (
            "invalid_base64_string!@#$",
            pytest.raises(ValueError, match="Invalid base64-encoded string"),
        ),
        (
            base64_encode_str("metadata:\n  pipeline_name: v1.5.1_pipeline\n  iris_version: 1.0.0\n\npipeline: []"),
            pytest.raises(IRISPipelineError),
        ),
        (
            base64_encode_str(
                f"metadata:\n  pipeline_name: v1.5.1_pipeline\n  iris_version: {__version__}\n\npipeline: [\n"
                f"  {{\n    'name': 'segmentation',\n    'algorithm': {{\n"
                f"      'class_name': 'iris.nodes.segmentation.onnx_multilabel_segmentation.ONNXMultilabelSegmentation',\n"
                f"      'params': {{'model_path': 'dummy.onnx'}},\n    }},\n    'inputs': [{{'name': 'image', 'source_node': 'input'}}],\n"
                f"    'callbacks': [],\n    'seed': None,\n  }}\n]"
            ),
            does_not_raise(),
        ),
        (
            base64_encode_str(load_config_str("incoherent_pipeline_1.yml")),
            pytest.raises(IRISPipelineError),
        ),
        (
            base64_encode_str(
                f"metadata:\n  pipeline_name: v1.5.1_pipeline\n  iris_version: {__version__}\n\npipeline: [\n"
                f"  {{\n    'name': 'segmentation',\n    'algorithm': {{\n"
                f"      'class_name': 'iris.nodes.segmentation.onnx_multilabel_segmentation.ONNXMultilabelSegmentation',\n"
                f"      'params': {{'model_path': 'dummy.onnx'}},\n    }},\n    'inputs': [{{'name': 'image', 'source_node': 'input'}}],\n"
                f"    'callbacks': [],\n    'seed': None,\n  }}\n]"
            ),
            does_not_raise(),
        ),
    ],
    ids=[
        "valid_empty_pipeline",
        "invalid_base64",
        "wrong_version",
        "duplicate_node_names",
        "incoherent_pipeline",
        "valid_pipeline",
    ],
)
def test_update_config(config: str, expectation):
    """Test the update_config method with various configurations."""
    pipeline = IRISPipeline()
    with expectation:
        mock_model = MagicMock()
        mock_model.SerializeToString.return_value = b"dummy_model_data"
        mock_session = MagicMock(spec=ort.InferenceSession)
        with patch("onnx.load", return_value=mock_model), patch("onnx.checker.check_model"), patch(
            "onnxruntime.InferenceSession", return_value=mock_session
        ):
            pipeline.update_config(config)
            if isinstance(expectation, does_not_raise):
                assert isinstance(pipeline.nodes, dict)
                if config != "invalid_base64_string!@#$":
                    assert pipeline.params.metadata.pipeline_name == "v1.5.1_pipeline"


def test_update_config_preserves_environment():
    """Test that update_config preserves the pipeline environment."""
    env = Environment(
        pipeline_output_builder=build_simple_iris_pipeline_debugging_output,
        error_manager=store_error_manager,
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )
    pipeline = IRISPipeline(env=env)

    # Update config
    config = base64_encode_str(
        f"metadata:\n  pipeline_name: test_pipeline\n  iris_version: {__version__}\n\npipeline: []"
    )
    pipeline.update_config(config)

    # Verify environment is preserved
    assert pipeline.env.pipeline_output_builder == build_simple_iris_pipeline_debugging_output
    assert pipeline.env.error_manager == store_error_manager
    assert pipeline.env.call_trace_initialiser == PipelineCallTraceStorage.initialise


def test_update_config_reinitializes_nodes():
    """Test that update_config properly reinitializes nodes and call trace."""
    pipeline = IRISPipeline()

    # Get initial state
    initial_nodes = pipeline.nodes
    initial_call_trace = pipeline.call_trace

    # Update config
    config = base64_encode_str(
        f"metadata:\n  pipeline_name: test_pipeline\n  iris_version: {__version__}\n\npipeline: []"
    )
    pipeline.update_config(config)

    # Verify nodes and call trace were reinitialized
    assert pipeline.nodes is not initial_nodes
    assert pipeline.call_trace is not initial_call_trace
    assert isinstance(pipeline.nodes, dict)


@pytest.mark.parametrize(
    "input_version,expected_version,should_pass",
    [
        ("1.0.0", "1.0.0", True),
        ("2.5.1", "2.5.1", True),
        ("1.0.0", "2.0.0", False),
        ("1.6.1", "2.0.0", False),
        ("0.1.0-alpha", "0.1.0-alpha", True),
    ],
    ids=[
        "matching_basic_semver",
        "matching_different_version",
        "mismatched_major",
        "mismatched_specific",
        "matching_alpha_version",
    ],
)
def test_version_validation_behavior(input_version, expected_version, should_pass):
    """Test version validation behavior with various version combinations."""

    class CustomPipeline(IRISPipeline):
        pass
        # PACKAGE_VERSION = expected_version

    # monkey patch the version
    with patch.object(CustomPipeline, "PACKAGE_VERSION", expected_version):
        config = {
            "metadata": {
                "pipeline_name": "custom_pipeline",
                "iris_version": input_version,
            },
            "pipeline": [],
        }

        if should_pass:
            # Should create pipeline without error
            try:
                pipeline = CustomPipeline(config=config)
                assert pipeline.params.metadata.iris_version == input_version
            except IRISPipelineError:
                pytest.fail(f"Expected version {input_version} to match {expected_version}")
        else:
            # Should raise version mismatch error
            with pytest.raises(IRISPipelineError) as exc_info:
                CustomPipeline(config=config)

            error_message = str(exc_info.value)
            assert "Wrong config version" in error_message
            assert expected_version in error_message
            assert input_version in error_message

    # del CustomPipeline
