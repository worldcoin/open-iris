import os
import pickle
from copy import deepcopy
from typing import Any

import pytest

from iris.nodes.matcher.hamming_distance_matcher import HashBasedMatcher
from iris.io.dataclasses import IrisTemplate


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "hamming_distance_matcher")
    mock_path = os.path.join(testdir, f"{name}.pickle")
    obj = pickle.load(open(mock_path, "rb"))
    if isinstance(obj, IrisTemplate) and not hasattr(obj, "iris_code_version"):
        obj = IrisTemplate(
            iris_codes=obj.iris_codes,
            mask_codes=obj.mask_codes,
            iris_code_version="v1.0",
        )
    return obj


@pytest.mark.parametrize(
    "rotation_shift,hash_bits,expected_result",
    [
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
        pytest.param(0, 40, 0.0),  # Same template should match exactly
    ],
    ids=[
        "hash_based_1",
        "hash_based_2",
        "hash_based_3",
        "hash_based_4",
        "hash_based_5",
        "hash_based_6",
        "hash_based_7",
        "hash_based_8",
        "hash_based_9",
        "hash_based_10",
    ],
)
def test_e2e_hash_based_matcher(
    rotation_shift: int,
    hash_bits: int,
    expected_result: float,
) -> None:
    first_template = load_mock_pickle("iris_template")
    second_template = deepcopy(first_template)

    matcher = HashBasedMatcher(
        rotation_shift=rotation_shift,
        hash_bits=hash_bits,
    )
    result = matcher.run(first_template, second_template)

    assert round(result, 4) == expected_result
