from typing import Tuple

import pytest
from pydantic import ValidationError, confloat

from iris.nodes.iris_response_refinement.fragile_bits_refinement import FragileBitRefinement, FragileType


@pytest.mark.parametrize(
    # given
    "value_threshold, fragile_type, maskisduplicated",
    [
        pytest.param([-0.6, 0, -0.3], FragileType.cartesian, 1),
        pytest.param([-0.2, -0.5, 1], FragileType.polar, 0),
        pytest.param([-0.2, -0.5, 1], FragileType.polar, 'a'),
        pytest.param([-0.2, -0.5, 1], FragileType.polar, 0.5),
        pytest.param([0, 0, 1], "elliptical", 1),
    ],
    ids=["error_theshold_cartesian", "error_theshold_polar", "error_bool1", "error_bool2", "error_fragile_type"],
)
def test_iris_encoder_threshold_raises_an_exception(
    value_threshold: Tuple[confloat(ge=0), confloat(ge=0), confloat(ge=0)],
    fragile_type: FragileType,
    maskisduplicated: bool,
) -> None:
    # when
    with pytest.raises((ValidationError)):
        _ = FragileBitRefinement(value_threshold, fragile_type, maskisduplicated)
