import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import Landmarks
from iris.orchestration.output_builders import build_debugging_output, build_orb_output, build_simple_output
from tests.e2e_tests.utils import (
    compare_debug_pipeline_outputs,
    compare_iris_pipeline_outputs,
    compare_simple_pipeline_outputs,
)


@pytest.fixture
def expected_simple_iris_pipeline_output() -> Tuple[Tuple[np.ndarray, np.ndarray], Landmarks, Dict[str, Any]]:
    expected_output_path = os.path.join(os.path.dirname(__file__), "mocks", "expected_iris_pipeline_simple_output.pickle")
    return pickle.load(open(expected_output_path, "rb"))


@pytest.fixture
def expected_orb_iris_pipeline_output() -> Tuple[Tuple[np.ndarray, np.ndarray], Landmarks, Dict[str, Any]]:
    expected_output_path = os.path.join(os.path.dirname(__file__), "mocks", "expected_iris_pipeline_orb_output.pickle")
    return pickle.load(open(expected_output_path, "rb"))


@pytest.fixture
def expected_debug_iris_pipeline_output() -> Any:
    expected_output_path = os.path.join(
        os.path.dirname(__file__), "mocks", "expected_iris_pipeline_debug_output.pickle"
    )
    return pickle.load(open(expected_output_path, "rb"))


@pytest.fixture
def mock_iris_pipeline_call_trace() -> PipelineCallTraceStorage:
    expected_call_trace_path = os.path.join(os.path.dirname(__file__), "mocks", "mock_iris_pipeline_call_trace.pickle")

    return pickle.load(open(expected_call_trace_path, "rb"))


def test_e2e_build_simple_output(
    mock_iris_pipeline_call_trace: PipelineCallTraceStorage,
    expected_simple_iris_pipeline_output: Tuple[Tuple[np.ndarray, np.ndarray], Landmarks, Dict[str, Any]],
) -> None:
    build_orb_iris_pipeline_output = build_simple_output(mock_iris_pipeline_call_trace)

    compare_simple_pipeline_outputs(expected_simple_iris_pipeline_output, build_orb_iris_pipeline_output)


def test_e2e_build_orb_output(
    mock_iris_pipeline_call_trace: PipelineCallTraceStorage,
    expected_orb_iris_pipeline_output: Tuple[Tuple[np.ndarray, np.ndarray], Landmarks, Dict[str, Any]],
) -> None:
    build_orb_iris_pipeline_output = build_orb_output(mock_iris_pipeline_call_trace)

    compare_iris_pipeline_outputs(expected_orb_iris_pipeline_output, build_orb_iris_pipeline_output)


def test_e2e_build_debug_output(
    mock_iris_pipeline_call_trace: PipelineCallTraceStorage,
    expected_debug_iris_pipeline_output: Tuple[Tuple[np.ndarray, np.ndarray], Landmarks, Dict[str, Any]],
) -> None:
    build_debug_iris_pipeline_output = build_debugging_output(mock_iris_pipeline_call_trace)

    compare_debug_pipeline_outputs(expected_debug_iris_pipeline_output, build_debug_iris_pipeline_output)
