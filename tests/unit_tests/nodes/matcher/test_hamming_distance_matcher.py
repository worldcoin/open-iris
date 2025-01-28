from typing import List

import numpy as np
import pytest
from pydantic import ValidationError

from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher


@pytest.mark.parametrize(
    "rotation_shift, norm_mean",
    [
        pytest.param(-0.5, 0.45),
        pytest.param(1.5, None),
        pytest.param(200, "a"),
        pytest.param(100, -0.2),
        pytest.param(10, 1.3),
    ],
    ids=[
        "rotation_shift should not be negative",
        "rotation_shift should not be floating points",
        "norm_mean should be float",
        "norm_mean should not be negative",
        "norm_mean should not be more than 1",
    ],
)
def test_iris_matcher_raises_an_exception1(
    rotation_shift: int,
    norm_mean: bool,
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(rotation_shift=rotation_shift, norm_mean=norm_mean)


@pytest.mark.parametrize(
    "rotation_shift, norm_mean, weights",
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
    norm_mean: float,
    weights: List[np.ndarray],
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(rotation_shift=rotation_shift, norm_mean=norm_mean, weights=weights)


@pytest.mark.parametrize(
    "rotation_shift, norm_mean, norm_gradient, separate_half_matching, weights",
    [
        pytest.param(5, 0.4, "a", False, 3),
        pytest.param(15, None, 0.0005, "b", np.zeros((3, 4))),
    ],
    ids=[
        "norm_gradient should be float",
        "separate_half_matching should be bool",
    ],
)
def test_iris_matcher_raises_an_exception2(
    rotation_shift: int,
    norm_mean: float,
    norm_gradient: float,
    separate_half_matching: bool,
    weights: List[np.ndarray],
) -> None:
    with pytest.raises(ValidationError):
        _ = HammingDistanceMatcher(
            rotation_shift,
            norm_mean=norm_mean,
            norm_gradient=norm_gradient,
            separate_half_matching=separate_half_matching,
            weights=weights,
        )
