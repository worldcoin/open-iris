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
    "rotation_shift,normalise,norm_mean,norm_gradient,separate_half_matching,weights,expected_result",
    [
        pytest.param(10, False, 0.45, 0.00005, True, None, 0.0),
        pytest.param(15, False, 0.45, 0.00005, False, None, 0.0),
        pytest.param(10, True, 0.45, 0.00005, True, None, 0.0347),
        pytest.param(15, True, 0.45, 0.00005, False, None, 0),
        pytest.param(10, False, 0.45, 0.00005, True, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0),
        pytest.param(15, False, 0.45, 0.00005, False, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0),
        pytest.param(10, True, 0.45, 0.00005, True, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0347),
        pytest.param(15, True, 0.45, 0.00005, False, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0),
        pytest.param(10, True, 0.45, 0.001, True, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0),
        pytest.param(15, True, 0.45, 0.00008, False, [np.ones([16, 256, 2]), np.ones([16, 256, 2])], 0.0),
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
        "regular_normalizedweighted1_normgradient1",
        "regular_normalizedweighted2_normgradient2",
    ],
)
def test_e2e_iris_matcher(
    rotation_shift: int,
    normalise: bool,
    norm_mean: float,
    norm_gradient: float,
    separate_half_matching: bool,
    weights: Optional[List[np.ndarray]],
    expected_result: float,
) -> None:
    first_template = load_mock_pickle("iris_template")
    second_template = deepcopy(first_template)

    matcher = HammingDistanceMatcher(
        rotation_shift=rotation_shift,
        normalise=normalise,
        norm_mean=norm_mean,
        norm_gradient=norm_gradient,
        separate_half_matching=separate_half_matching,
        weights=weights,
    )
    result = matcher.run(first_template, second_template)

    assert round(result, 4) == expected_result
