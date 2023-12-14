from typing import List

import numpy as np
import pytest
from _pytest.fixtures import FixtureRequest
from pydantic import ValidationError

from iris.io.errors import ProbeSchemaError
from iris.nodes.iris_response.conv_filter_bank import ConvFilterBank
from iris.nodes.iris_response.image_filters import gabor_filters
from iris.nodes.iris_response.image_filters.image_filter_interface import ImageFilter
from iris.nodes.iris_response.probe_schemas.probe_schema_interface import ProbeSchema
from iris.nodes.iris_response.probe_schemas.regular_probe_schema import RegularProbeSchema


@pytest.fixture
def filter_1() -> ImageFilter:
    return gabor_filters.GaborFilter(
        kernel_size=(21, 21), sigma_phi=2, sigma_rho=4, theta_degrees=45, lambda_phi=10, dc_correction=True
    )


@pytest.fixture
def filter_2() -> ImageFilter:
    return gabor_filters.GaborFilter(
        kernel_size=(15, 17), sigma_phi=1.5, sigma_rho=2.5, theta_degrees=90, lambda_phi=8, dc_correction=True
    )


@pytest.fixture
def filter_3() -> ImageFilter:
    return gabor_filters.LogGaborFilter(
        kernel_size=(19, 17), sigma_phi=np.pi / 10, sigma_rho=0.8, theta_degrees=5.5, lambda_rho=8.5
    )


@pytest.fixture
def probeschema_1() -> ProbeSchema:
    return RegularProbeSchema(n_rows=25, n_cols=100)


@pytest.fixture
def probeschema_2() -> ProbeSchema:
    return RegularProbeSchema(n_rows=100, n_cols=300)


@pytest.fixture
def probeschema_3() -> ProbeSchema:
    return RegularProbeSchema(n_rows=60, n_cols=125)


@pytest.mark.parametrize(
    "filters, probe_schemas",
    [
        pytest.param(
            ["filter_1", "filter_2"],
            ["probeschema_1", "probeschema_2", "probeschema_3"],
        ),
        pytest.param(
            ["filter_1", "filter_2", "filter_3"],
            ["probeschema_1", "probeschema_2"],
        ),
    ],
    ids=[
        "filters and probe_schemas are not of the same length",
        "filters and probe_schemas are not of the same length again",
    ],
)
def test_convfilterbank_constructor_raises_an_exception_1(
    filters: List[ImageFilter],
    probe_schemas: List[ProbeSchema],
    request: FixtureRequest,
) -> None:
    loaded_filters = [request.getfixturevalue(img_filter) for img_filter in filters]
    loaded_probe_schemas = [request.getfixturevalue(probe_schema) for probe_schema in probe_schemas]

    with pytest.raises(ValidationError):
        _ = ConvFilterBank(
            filters=loaded_filters,
            probe_schemas=loaded_probe_schemas,
        )


@pytest.mark.parametrize(
    "filters, probe_schemas",
    [
        pytest.param(
            [[np.zeros((21, 21)), np.ones((21, 21))]],
            ["probeschema_1", "probeschema_2"],
        ),
        pytest.param(
            ["a", "b"],
            ["probeschema_1", "probeschema_2"],
        ),
        pytest.param(
            [],
            ["probeschema_1"],
        ),
    ],
    ids=[
        "filters is not of List[ImageFilter] type",
        "filters is not of List[ImageFilter] type again",
        "filters is empty",
    ],
)
def test_convfilterbank_constructor_raises_an_exception_2(
    filters: List[ImageFilter],
    probe_schemas: List[ProbeSchema],
    request: FixtureRequest,
) -> None:
    loaded_probe_schemas = [request.getfixturevalue(probe_schema) for probe_schema in probe_schemas]

    with pytest.raises(ValidationError):
        _ = ConvFilterBank(
            filters=filters,
            probe_schemas=loaded_probe_schemas,
        )


@pytest.mark.parametrize(
    "filters, probe_schemas",
    [
        pytest.param(
            ["filter_1", "filter_2"],
            [[np.zeros((21, 21)), np.ones((21, 21))]],
        ),
        pytest.param(
            ["filter_1", "filter_2"],
            ["a", "b"],
        ),
        pytest.param(
            ["filter_1"],
            [],
        ),
    ],
    ids=[
        "probe_schemas is not of List[probe_schema] type",
        "probe_schemas is not of List[probe_schema] type again",
        "probe_schemas is empty",
    ],
)
def test_convfilterbank_constructor_raises_an_exception_3(
    filters: List[ImageFilter],
    probe_schemas: List[ProbeSchema],
    request: FixtureRequest,
) -> None:
    loaded_filters = [request.getfixturevalue(img_filter) for img_filter in filters]

    with pytest.raises((ValidationError, ProbeSchemaError)):
        _ = ConvFilterBank(
            filters=loaded_filters,
            probe_schemas=probe_schemas,
        )
