import os
import pickle

import numpy as np
import pytest

from iris.nodes.iris_response.image_filters import gabor_filters


@pytest.fixture
def precomputed_filters_dirpath() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "mocks")


def test_computed_kernel_values(precomputed_filters_dirpath: str) -> None:
    filename = os.path.join(precomputed_filters_dirpath, "image_filters", "gabor_filter.pickle")
    expected_gabor_filter = pickle.load(open(filename, "rb"))

    filename = os.path.join(precomputed_filters_dirpath, "image_filters", "gabor2_filter.pickle")
    expected_gabor2_filter = pickle.load(open(filename, "rb"))

    filename = os.path.join(precomputed_filters_dirpath, "image_filters", "loggabor_filter.pickle")
    expected_loggabor_filter = pickle.load(open(filename, "rb"))

    first_result = gabor_filters.GaborFilter(
        kernel_size=(21, 21), sigma_phi=2, sigma_rho=4, theta_degrees=45, lambda_phi=10, dc_correction=True
    )
    second_result = gabor_filters.GaborFilter(
        kernel_size=(15, 17), sigma_phi=1.5, sigma_rho=2.5, theta_degrees=90, lambda_phi=8, dc_correction=True
    )
    third_result = gabor_filters.LogGaborFilter(
        kernel_size=(19, 17), sigma_phi=np.pi / 10, sigma_rho=0.8, theta_degrees=5.5, lambda_rho=8.5
    )

    assert np.allclose(expected_gabor_filter, first_result.kernel_values, rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_gabor2_filter, second_result.kernel_values, rtol=1e-05, atol=1e-07)
    assert np.allclose(expected_loggabor_filter, third_result.kernel_values, rtol=1e-05, atol=1e-07)
