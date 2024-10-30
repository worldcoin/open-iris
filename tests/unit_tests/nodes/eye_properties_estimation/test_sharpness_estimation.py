from typing import Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from iris.nodes.eye_properties_estimation.sharpness_estimation import SharpnessEstimation


@pytest.mark.parametrize(
    "lap_ksize",
    [
        pytest.param(0),
        pytest.param("a"),
        pytest.param(-10),
        pytest.param(33),
        pytest.param(2),
        pytest.param(np.ones(3)),
    ],
    ids=[
        "lap_ksize should be larger than zero",
        "lap_ksize should be int",
        "lap_ksize should not be negative",
        "lap_ksize should not be larger than 31",
        "lap_ksize should be odd number",
        "lap_ksize should not be array",
    ],
)
def test_sharpness_lap_ksize_raises_an_exception(lap_ksize: int) -> None:
    with pytest.raises(ValidationError):
        _ = SharpnessEstimation(lap_ksize=lap_ksize)


@pytest.mark.parametrize(
    "erosion_ksize",
    [
        pytest.param((0, 5)),
        pytest.param((1, "a")),
        pytest.param((-10, 3)),
        pytest.param((30, 5)),
        pytest.param(np.ones(3)),
    ],
    ids=[
        "erosion_ksize should all be larger than zero",
        "erosion_ksize should all be int",
        "erosion_ksize should not be negative",
        "erosion_ksize should be odd number",
        "erosion_ksize should be a tuple of integer with length 2",
    ],
)
def test_sharpness_erosion_ksize_raises_an_exception(erosion_ksize: Tuple[int, int]) -> None:
    with pytest.raises(ValidationError):
        _ = SharpnessEstimation(erosion_ksize=erosion_ksize)
