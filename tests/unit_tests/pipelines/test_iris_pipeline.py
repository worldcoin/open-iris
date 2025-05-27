import os
from contextlib import nullcontext as does_not_raise
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
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
            pytest.raises(IRISPipelineError),
        ),
        (
            {"not": ["a", "str"]},
            pytest.raises(IRISPipelineError),
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

    node = iris_pipeline.instanciate_node(node_class=node_class, algorithm_params=algorithm_params, callbacks=callbacks)

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

    nodes = iris_pipeline.instanciate_nodes()

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
