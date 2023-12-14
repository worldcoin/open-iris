import os
import pickle
from copy import deepcopy
from typing import Any, List, Optional

import numpy as np
import pytest

from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "hamming_distance_matcher")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.mark.parametrize(
    "rotation_shift,nm_dist,weights,expected_result",
    [
        pytest.param(10, None, None, 0.0),
        pytest.param(15, None, None, 0.0),
        pytest.param(10, 0.45, None, 0.0123),
        pytest.param(15, 0.45, None, 0.0123),
        pytest.param(10, None, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0),
        pytest.param(15, None, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0),
        pytest.param(10, 0.45, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0492),
        pytest.param(15, 0.45, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0492),
    ],
    ids=[
        "regular1",
        "regular2",
        "regular_normalized1",
        "regular_normalized2",
        "regular_weighted1",
        "regular_weighted2",
        "regular_normalizedweighted1",
        "regular_normalizedweighted2",
    ],
)
def test_e2e_iris_matcher(
    rotation_shift: int,
    nm_dist: float,
    weights: Optional[List[np.ndarray]],
    expected_result: float,
) -> None:
    first_template = load_mock_pickle("iris_template")
    second_template = deepcopy(first_template)

    matcher = HammingDistanceMatcher(rotation_shift, nm_dist, weights)
    result = matcher.run(first_template, second_template)

    assert round(result, 4) == expected_result
