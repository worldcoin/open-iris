import os
import pickle
from typing import Any

import numpy as np

from iris.nodes.encoder.iris_encoder import IrisEncoder


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "iris_encoder")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


def test_iris_encoder_constructor() -> None:
    iris_response = load_mock_pickle("iris_response")
    expected_result = load_mock_pickle("e2e_expected_result")

    iris_encoder = IrisEncoder(mask_threshold=0.5)
    result = iris_encoder(iris_response)

    assert len(result.iris_codes) == len(expected_result.iris_codes)
    assert len(result.mask_codes) == len(expected_result.mask_codes)

    for i, (i_iris_code, i_mask_code) in enumerate(zip(result.iris_codes, result.mask_codes)):
        assert np.allclose(expected_result.iris_codes[i], i_iris_code, rtol=1e-05, atol=1e-07)
        assert np.allclose(expected_result.mask_codes[i], i_mask_code, rtol=1e-05, atol=1e-07)
