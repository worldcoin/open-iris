from typing import List

import numpy as np
import pytest
from pydantic import ValidationError

from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher


@pytest.mark.parametrize(
    "rotation_shift, nm_dist",
    [
        pytest.param(-0.5, 0.45),
        pytest.param(1.5, None),
        pytest.param(200, 0.3),
        pytest.param(200, "a"),
        pytest.param(100, -0.2),
        pytest.param(10, 1.3),
    ],
    ids=[
        "rotation_shift should not be negative",
        "rotation_shift should not be floating points",
        "rotation_shift should not be larger than 180",
        "nm_dist should be float",
        "nm_dist should not be negative",
        "nm_dist should not be more than 1",
    ],
)
def test_iris_matcher_raises_an_exception1(
    rotation_shift: int,
    nm_dist: bool,
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(rotation_shift, nm_dist)


@pytest.mark.parametrize(
    "rotation_shift, nm_dist, weights",
    [
        pytest.param(5, 0.4, 3),
        pytest.param(15, None, np.zeros((3, 4))),
        pytest.param(200, 0.45, [("a", 13)]),
    ],
    ids=[
        "weights should be a list of arrays",
        "weights should be a list of arrays",
        "n_rows need to be int or float",
    ],
)
def test_iris_matcher_raises_an_exception2(
    rotation_shift: int,
    nm_dist: float,
    weights: List[np.ndarray],
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(rotation_shift, nm_dist, weights)
