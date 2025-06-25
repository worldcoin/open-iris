import numpy as np
import pytest

from iris.nodes.iris_response.image_filters.image_filter_interface import ImageFilter, ImageFilterError


# Mock ImageFilters implementation for testing mechanism purposes
class AverageFilter(ImageFilter):
    class AverageFilterParameters(ImageFilter.Parameters):
        weight: float

    __parameters_type__ = AverageFilterParameters

    def __init__(self, weight: float) -> None:
        super().__init__(weight=weight)

    def compute_kernel_values(self) -> np.ndarray:
        kernel_value = np.ones(shape=(3, 3)) * self.params.weight
        return kernel_value / np.linalg.norm(kernel_value, ord="fro")


def test_parameters_assignment() -> None:
    expected_param_class_name = "AverageFilterParameters"
    expected_num_params = 1
    expected_param_name = "weight"
    expected_param_type = float
    expected_param_value = 3.0

    mock_filter = AverageFilter(weight=3.0)
    filter_params = mock_filter.params

    assert filter_params.__class__.__name__ == expected_param_class_name
    assert len(filter_params.__dict__) == expected_num_params
    assert list(filter_params.__dict__.keys())[0] == expected_param_name
    assert isinstance(filter_params.weight, expected_param_type)
    assert filter_params.weight == expected_param_value


def test_setting_kernel_values_raise_an_error() -> None:
    mock_filter = AverageFilter(weight=3.0)
    expected_err_msg = "ImageFilter kernel_values are immutable."

    with pytest.raises(ImageFilterError) as e:
        mock_filter.kernel_values = np.ones(shape=(3, 3))

    assert str(e.value) == expected_err_msg
