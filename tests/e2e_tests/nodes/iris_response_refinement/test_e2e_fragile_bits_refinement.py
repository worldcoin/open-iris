import os
import pickle
from typing import Any, Literal, Tuple

import numpy as np
import pytest

from iris.nodes.iris_response_refinement.fragile_bits_refinement import FragileBitRefinement


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "fragile_bits")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.mark.parametrize(
    "value_threshold,fragile_type",
    [
        pytest.param([0.5, 0.5], "cartesian"),
        pytest.param([0.49, np.pi / 8], "polar"),
    ],
    ids=["cartesian", "polar"],
)
def test_fragile_bits_dummy_responses(
    value_threshold: Tuple[float, float], fragile_type: Literal["cartesian", "polar"]
) -> None:
    iris_filter_response = load_mock_pickle(f"artificial_iris_responses_{fragile_type}")

    mask_responses_refined = load_mock_pickle(f"artificial_mask_responses_{fragile_type}_expected_refinement")

    fragile_bit_refinement = FragileBitRefinement(value_threshold=value_threshold, fragile_type=fragile_type)
    iris_filter_response_refined = fragile_bit_refinement(iris_filter_response)

    for refined_mask, obtained_mask in zip(mask_responses_refined, iris_filter_response_refined.mask_responses):
        assert np.array_equal(refined_mask, obtained_mask)
