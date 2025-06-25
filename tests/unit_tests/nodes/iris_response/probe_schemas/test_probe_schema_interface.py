from typing import Tuple

import numpy as np
import pytest
from pydantic import Field

from iris.io.errors import ProbeSchemaError
from iris.nodes.iris_response.probe_schemas.probe_schema_interface import ProbeSchema


# Mock ProbeSchema implementation for testing mechanism purposes
class MockProbeSchema(ProbeSchema):
    class MockProbeSchemaParameters(ProbeSchema.ProbeSchemaParameters):
        """Default MockProbeSchema parameters."""

        n_rows: int = Field(..., gt=0)
        n_cols: int = Field(..., gt=0)

    params: MockProbeSchemaParameters
    __parameters_type__ = MockProbeSchemaParameters

    def __init__(self, n_rows: int, n_cols: int) -> None:
        super().__init__(n_rows=n_rows, n_cols=n_cols)

    def generate_schema(self) -> Tuple[np.ndarray, np.ndarray]:
        return np.linspace(0, 1, self.params.n_rows), np.linspace(0, 1, self.params.n_cols)


def test_parameters_assignment() -> None:
    expected_param_class_name = "MockProbeSchemaParameters"
    expected_num_params = 2
    expected_params_names = sorted(["n_rows", "n_cols"])
    expected_params_type = int
    expected_params_value = 10

    mock_filter = MockProbeSchema(n_rows=10, n_cols=10)
    filter_params = mock_filter.params

    assert filter_params.__class__.__name__ == expected_param_class_name
    assert len(filter_params.__dict__) == expected_num_params
    assert sorted(list(filter_params.__dict__.keys())) == expected_params_names
    assert isinstance(filter_params.n_rows, expected_params_type)
    assert isinstance(filter_params.n_cols, expected_params_type)
    assert filter_params.n_rows == filter_params.n_cols == expected_params_value


def test_setting_rhos_values_raise_an_error() -> None:
    mock_schema = MockProbeSchema(n_rows=10, n_cols=10)
    expected_err_msg = "ProbeSchema rhos values are immutable."

    with pytest.raises(ProbeSchemaError) as e:
        mock_schema.rhos = np.arange(10)

    assert str(e.value) == expected_err_msg


def test_setting_phis_values_raise_an_error() -> None:
    mock_schema = MockProbeSchema(n_rows=10, n_cols=10)
    expected_err_msg = "ProbeSchema phis values are immutable."

    with pytest.raises(ProbeSchemaError) as e:
        mock_schema.phis = np.arange(10)

    assert str(e.value) == expected_err_msg
