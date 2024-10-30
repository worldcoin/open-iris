import math
import os
import pickle
from typing import Any, Literal

import numpy as np
import pytest

from iris.io.dataclasses import NormalizedIris
from iris.nodes.eye_properties_estimation.sharpness_estimation import SharpnessEstimation


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "iris_response/mocks", "conv_filter_bank")
    mock_path = os.path.join(testdir, f"{name}.pickle")
    return pickle.load(open(mock_path, "rb"))


def test_sharpness_estimation() -> None:
    normalized_iris = load_mock_pickle(name="normalized_iris")

    sharpness_obj = SharpnessEstimation()
    sharpness = sharpness_obj(normalized_iris)

    assert math.isclose(sharpness.score, 880.9419555664062)

    sharpness_obj = SharpnessEstimation(lap_ksize=7)
    sharpness = sharpness_obj(normalized_iris)

    assert math.isclose(sharpness.score, 5.179013252258301)

    sharpness_obj = SharpnessEstimation(erosion_ksize=[13, 7])
    sharpness = sharpness_obj(normalized_iris)

    assert math.isclose(sharpness.score, 1013.1661376953125)
