import os
import pickle
from typing import Any, Dict

import cv2
import numpy as np
import pytest

from iris.io.dataclasses import IRImage
from iris.pipelines.iris_pipeline import IRISPipeline
from tests.e2e_tests.utils import compare_debug_pipeline_outputs, compare_iris_pipeline_outputs


@pytest.fixture
def ir_image() -> np.ndarray:
    ir_image_path = os.path.join(os.path.dirname(__file__), "mocks", "inputs", "anonymized.png")
    img_data = cv2.imread(ir_image_path, cv2.IMREAD_GRAYSCALE)
    return img_data


@pytest.fixture
def expected_iris_pipeline_output() -> Dict[str, Any]:
    expected_iris_code_path = os.path.join(
        os.path.dirname(__file__), "mocks", "outputs", "expected_iris_orb_pipeline_output.pickle"
    )
    return pickle.load(open(expected_iris_code_path, "rb"))


@pytest.fixture
def expected_debug_pipeline_output() -> Dict[str, Any]:
    expected_iris_code_path = os.path.join(
        os.path.dirname(__file__), "mocks", "outputs", "expected_iris_debug_pipeline_output.pickle"
    )
    return pickle.load(open(expected_iris_code_path, "rb"))


def test_e2e_iris_pipeline(ir_image: np.ndarray, expected_iris_pipeline_output: Dict[str, Any]) -> None:
    """End-to-end test of the IRISPipeline in the Orb setup"""
    iris_pipeline = IRISPipeline(env=IRISPipeline.ORB_ENVIRONMENT)
    computed_pipeline_output = iris_pipeline(IRImage(img_data=ir_image, image_id="image_id", eye_side="right"))

    compare_iris_pipeline_outputs(computed_pipeline_output, expected_iris_pipeline_output)


def test_e2e_debug_pipeline(ir_image: np.ndarray, expected_debug_pipeline_output: Dict[str, Any]) -> None:
    """End-to-end test of the IRISPipeline in the debug setup"""
    iris_pipeline = IRISPipeline(env=IRISPipeline.DEBUGGING_ENVIRONMENT)

    computed_pipeline_output = iris_pipeline(IRImage(img_data=ir_image, image_id="image_id", eye_side="right"))

    compare_debug_pipeline_outputs(computed_pipeline_output, expected_debug_pipeline_output)
