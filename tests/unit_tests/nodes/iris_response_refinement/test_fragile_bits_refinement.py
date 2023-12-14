from typing import Literal, Tuple

import pytest
from pydantic import ValidationError, confloat

from iris.nodes.iris_response_refinement.fragile_bits_refinement import FragileBitRefinement


@pytest.mark.parametrize(
    "value_threshold,fragile_type",
    [
        pytest.param([-0.6, -0.3], "cartesian"),
        pytest.param([-0.2, -0.5], "polar"),
        pytest.param([0, 0], "elliptical"),
    ],
    ids=["error_threshold_cartesian", "error_threshold_polar", "error_fragile_type"],
)
def test_iris_encoder_threshold_raises_an_exception(
    value_threshold: Tuple[confloat(ge=0), confloat(ge=0)], fragile_type: Literal["cartesian", "polar"]
) -> None:
    with pytest.raises(ValidationError):
        _ = FragileBitRefinement(value_threshold, fragile_type)
