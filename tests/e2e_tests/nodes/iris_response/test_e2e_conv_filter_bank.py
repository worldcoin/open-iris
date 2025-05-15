import os
import pickle
from typing import Any, List

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest

from iris.nodes.iris_response.conv_filter_bank import ConvFilterBank
from iris.nodes.iris_response.image_filters import gabor_filters
from iris.nodes.iris_response.image_filters.image_filter_interface import ImageFilter
from iris.nodes.iris_response.probe_schemas.probe_schema_interface import ProbeSchema
from iris.nodes.iris_response.probe_schemas.regular_probe_schema import RegularProbeSchema


def load_mock_pickle(name: str) -> Any:
    testdir = os.path.join(os.path.dirname(__file__), "mocks", "conv_filter_bank")

    mock_path = os.path.join(testdir, f"{name}.pickle")

    return pickle.load(open(mock_path, "rb"))


@pytest.fixture
def first_filter() -> ImageFilter:
    return gabor_filters.GaborFilter(
        kernel_size=(21, 21), sigma_phi=2, sigma_rho=4, theta_degrees=45, lambda_phi=10, dc_correction=True
    )


@pytest.fixture
def second_filter() -> ImageFilter:
    return gabor_filters.GaborFilter(
        kernel_size=(15, 17), sigma_phi=1.5, sigma_rho=2.5, theta_degrees=90, lambda_phi=8, dc_correction=True
    )


@pytest.fixture
def third_filter() -> ImageFilter:
    return gabor_filters.LogGaborFilter(
        kernel_size=(19, 17), sigma_phi=np.pi / 10, sigma_rho=0.8, theta_degrees=5.5, lambda_rho=8.5
    )


@pytest.fixture
def first_schema() -> ProbeSchema:
    return RegularProbeSchema(n_rows=25, n_cols=100)


@pytest.fixture
def second_schema() -> ProbeSchema:
    return RegularProbeSchema(n_rows=100, n_cols=300)


@pytest.fixture
def third_schema() -> ProbeSchema:
    return RegularProbeSchema(n_rows=60, n_cols=125)


def test_computed_responses_maskisduplicated(
    first_filter: ImageFilter,
    second_filter: ImageFilter,
    third_filter: ImageFilter,
    first_schema: ProbeSchema,
    second_schema: ProbeSchema,
    third_schema: ProbeSchema,
) -> None:
    expected_result = load_mock_pickle("e2e_expected_result")

    filterbank = ConvFilterBank(
        filters=[first_filter, second_filter, third_filter],
        probe_schemas=[first_schema, second_schema, third_schema],
    )
    result = filterbank(normalization_output=load_mock_pickle("normalized_iris"))

    assert np.allclose(expected_result.iris_responses[0], result.iris_responses[0], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.iris_responses[1], result.iris_responses[1], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.iris_responses[2], result.iris_responses[2], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.mask_responses[0], result.mask_responses[0], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.mask_responses[1], result.mask_responses[1], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.mask_responses[2], result.mask_responses[2], rtol=1e-05, atol=1e-07)
    assert result.iris_code_version == "v0.1"

def test_computed_responses(
    first_filter: ImageFilter,
    second_filter: ImageFilter,
    third_filter: ImageFilter,
    first_schema: ProbeSchema,
    second_schema: ProbeSchema,
    third_schema: ProbeSchema,
) -> None:
    expected_result = load_mock_pickle("e2e_expected_result_masknotduplicated")

    filterbank = ConvFilterBank(
        filters=[first_filter, second_filter, third_filter],
        probe_schemas=[first_schema, second_schema, third_schema],
        maskisduplicated=False,
    )
    result = filterbank(normalization_output=load_mock_pickle("normalized_iris"))

    assert np.allclose(expected_result.iris_responses[0], result.iris_responses[0], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.iris_responses[1], result.iris_responses[1], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.iris_responses[2], result.iris_responses[2], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.mask_responses[0], result.mask_responses[0], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.mask_responses[1], result.mask_responses[1], rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_result.mask_responses[2], result.mask_responses[2], rtol=1e-05, atol=1e-07)
    assert result.iris_code_version == "v0.1"

@pytest.mark.parametrize(
    "filters, probe_schemas, maskisduplicated",
    [
        pytest.param(
            ["first_filter"],
            ["first_schema"],
            True,
        ),
        pytest.param(
            ["first_filter", "first_filter"],
            ["first_schema", "second_schema"],
            True,
        ),
        pytest.param(
            ["first_filter", "second_filter"],
            ["first_schema", "first_schema"],
            True,
        ),
        pytest.param(
            ["first_filter", "second_filter"],
            ["first_schema", "first_schema"],
            True,
        ),
        pytest.param(
            ["first_filter", "second_filter", "third_filter"],
            ["first_schema", "second_schema", "third_schema"],
            False,
        ),
    ],
    ids=["regular1", "regular2", "regular2_0", "regular3", "regular4"],
)
def test_convfilterbank_constructor(
    filters: List[ImageFilter], probe_schemas: List[ProbeSchema], maskisduplicated: bool, request: FixtureRequest
) -> None:
    loaded_filters = [request.getfixturevalue(img_filter) for img_filter in filters]
    loaded_probe_schemas = [request.getfixturevalue(probe_schema) for probe_schema in probe_schemas]

    assert len(filters) == len(probe_schemas)

    filterbank = ConvFilterBank(filters=loaded_filters, probe_schemas=loaded_probe_schemas, maskisduplicated=maskisduplicated)
    filter_responses = filterbank(normalization_output=load_mock_pickle("normalized_iris"))

    for i_iris_response, i_mask_response in zip(filter_responses.iris_responses, filter_responses.mask_responses):
        assert i_iris_response.shape == i_mask_response.shape
        assert np.iscomplexobj(i_iris_response)
        assert np.iscomplexobj(i_mask_response)
        assert i_mask_response.real.max() <= 1
        assert i_mask_response.real.min() >= 0
        assert i_mask_response.imag.max() <= 1
        assert i_mask_response.imag.min() >= 0
        assert i_mask_response.real.max() > i_mask_response.real.min()
        assert i_mask_response.imag.max() > i_mask_response.imag.min()
        assert i_iris_response.real.max() > i_iris_response.real.min()
        assert i_iris_response.imag.max() > i_iris_response.imag.min()
    assert filter_responses.iris_code_version == "v0.1"
