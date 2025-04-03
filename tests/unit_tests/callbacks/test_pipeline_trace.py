import copy
from typing import Any, Dict, List

import pytest
from _pytest.fixtures import FixtureRequest

from iris.callbacks.pipeline_trace import NodeResultsWriter, PipelineCallTraceStorage, PipelineCallTraceStorageError
from iris.io.class_configs import Algorithm
from iris.nodes.eye_properties_estimation.occlusion_calculator import OcclusionCalculator
from iris.nodes.validators.object_validators import OcclusionValidator
from iris.orchestration.pipeline_dataclasses import PipelineClass, PipelineInput, PipelineNode


@pytest.fixture
def mock_alg_names() -> List[str]:
    return ["alg1", "alg2", "alg3", "alg4"]


@pytest.fixture
def mock_nodes() -> Dict[str, Algorithm]:
    return {
        "alg1": OcclusionCalculator(quantile_angle=0.1),
        "alg2": OcclusionCalculator(quantile_angle=0.7, callbacks=[OcclusionValidator(min_allowed_occlusion=0.3)]),
        "alg3": OcclusionCalculator(quantile_angle=0.3),
        "alg4": OcclusionCalculator(quantile_angle=0.5),
    }


@pytest.fixture
def mock_pipeline() -> Dict[str, PipelineNode]:
    return _create_mock_pipeline()


@pytest.fixture
def create_mock_pipeline(request):
    return _create_mock_pipeline(**request.param)


def _create_mock_pipeline(
    alg1_params: Dict[str, Any] = {"quantile_angle": 0.1},
    alg2_params: Dict[str, Any] = {"quantile_angle": 0.7},
    alg2_callback_params: Dict[str, Any] = {"min_allowed_occlusion": 0.3},
    alg3_params: Dict[str, Any] = {"quantile_angle": 0.3},
    alg4_params: Dict[str, Any] = {"quantile_angle": 0.5},
):
    return [
        PipelineNode(
            name="alg1",
            algorithm=PipelineClass(
                class_name="iris.nodes.eye_properties_estimation.occlusion_calculator.OcclusionCalculator",
                params=alg1_params,
            ),
            inputs=[PipelineInput(name="xx", source_node="input")],
        ),
        PipelineNode(
            name="alg2",
            algorithm=PipelineClass(
                class_name="iris.nodes.eye_properties_estimation.occlusion_calculator.OcclusionCalculator",
                params=alg2_params,
            ),
            inputs=[PipelineInput(name="xx", source_node="input"), PipelineInput(name="yy", source_node="alg1")],
            callbacks=[
                PipelineClass(
                    class_name="iris.nodes.validators.object_validators.OcclusionValidator",
                    params=alg2_callback_params,
                )
            ],
        ),
        PipelineNode(
            name="alg3",
            algorithm=PipelineClass(
                class_name="iris.nodes.eye_properties_estimation.occlusion_calculator.OcclusionCalculator",
                params=alg3_params,
            ),
            inputs=[PipelineInput(name="xx", source_node="alg2")],
        ),
        PipelineNode(
            name="alg4",
            algorithm=PipelineClass(
                class_name="iris.nodes.eye_properties_estimation.occlusion_calculator.OcclusionCalculator",
                params=alg4_params,
            ),
            inputs=[PipelineInput(name="xx", source_node="alg2"), PipelineInput(name="xx", source_node="alg3")],
        ),
    ]


@pytest.fixture
def call_trace_storage(mock_alg_names: List[str]) -> PipelineCallTraceStorage:
    return PipelineCallTraceStorage(results_names=mock_alg_names)


@pytest.mark.parametrize(
    "storage",
    [("call_trace_storage")],
    ids=["PipelineCallTraceStorage"],
)
def test_init_storage(storage: PipelineCallTraceStorage, mock_alg_names: List[str], request: FixtureRequest) -> None:
    storage = request.getfixturevalue(storage)
    assert sorted(storage._storage.keys()) == sorted(
        [*mock_alg_names, PipelineCallTraceStorage.INPUT_KEY_NAME, PipelineCallTraceStorage.ERROR_KEY_NAME]
    )
    assert list(storage._storage.values()) == [None] * len(storage)


@pytest.mark.parametrize(
    "storage",
    [("call_trace_storage")],
    ids=["PipelineCallTraceStorage"],
)
def test_write_get_input(storage: PipelineCallTraceStorage, request: FixtureRequest) -> None:
    storage = request.getfixturevalue(storage)
    mock_result = "mock_result"

    storage.write_input(mock_result)

    assert storage.get_input() == mock_result


@pytest.mark.parametrize(
    "storage",
    [("call_trace_storage")],
    ids=["PipelineCallTraceStorage"],
)
def test_write_get_error(storage: PipelineCallTraceStorage, request: FixtureRequest) -> None:
    storage = request.getfixturevalue(storage)

    assert storage.get_error() is None

    mock_error = ValueError("test")
    storage.write_error(mock_error)

    assert storage.get_error() == mock_error


@pytest.mark.parametrize(
    "storage",
    [("call_trace_storage")],
    ids=["PipelineCallTraceStorage"],
)
def test_write_get_parameter(storage: PipelineCallTraceStorage, request: FixtureRequest) -> None:
    storage = request.getfixturevalue(storage)
    storage.write_input("")

    mock_result = "mock_result"

    storage.write("alg1", mock_result)
    storage.write("alg4", mock_result)

    assert storage.get("alg1") == mock_result
    assert storage["alg2"] is None
    assert storage["alg3"] is None
    assert storage["alg4"] == mock_result


@pytest.mark.parametrize(
    "storage",
    [("call_trace_storage")],
    ids=["PipelineCallTraceStorage"],
)
def test_clean(storage: PipelineCallTraceStorage, mock_alg_names: List[str], request: FixtureRequest) -> None:
    storage = request.getfixturevalue(storage)
    storage.write_input("")
    mock_result = "mock_result"

    storage.write("alg1", mock_result)
    storage.write("alg4", mock_result)
    assert storage.get("alg1") == mock_result
    assert storage["alg4"] == mock_result

    storage.clean()

    assert sorted(storage._storage.keys()) == sorted(
        [*mock_alg_names, PipelineCallTraceStorage.INPUT_KEY_NAME, PipelineCallTraceStorage.ERROR_KEY_NAME]
    )
    assert list(storage._storage.values()) == [None] * len(storage)


@pytest.mark.parametrize(
    "storage",
    [("call_trace_storage")],
    ids=["PipelineCallTraceStorage"],
)
def test_node_result_writer(storage: PipelineCallTraceStorage, request: FixtureRequest) -> None:
    storage = request.getfixturevalue(storage)
    storage.write_input("")

    node_result_cb = NodeResultsWriter(trace_storage_reference=storage, result_name="alg1")
    mock_result = "mock_result"

    node_result_cb.on_execute_end(mock_result)
    assert storage.get("alg1") == mock_result


def test_pipeline_call_trace_storage_initialise(
    mock_nodes: Dict[str, Algorithm], mock_pipeline: Dict[str, PipelineNode]
) -> None:
    mock_nodes = copy.deepcopy(mock_nodes)
    call_trace_storage = PipelineCallTraceStorage.initialise(mock_nodes, mock_pipeline)

    assert set(mock_nodes.keys()).issubset(set(call_trace_storage._storage.keys()))
    for node in mock_nodes.values():
        assert any([isinstance(cb, NodeResultsWriter) for cb in node._callbacks])
