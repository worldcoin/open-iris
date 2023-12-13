import math
from typing import Tuple

import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import ImageFilterError
from iris.nodes.iris_response.image_filters import gabor_filters


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_phi, dc_correction",
    [
        ((19, 21), 2, 4, 45, 10, True),
        ((11, 15), 1.1, 1.5, 0, 3, True),
        ((31, 31), 1, 1, 240.7, 2, True),
    ],
    ids=[
        "regular1",
        "regular2",
        "regular3",
    ],
)
def test_gabor_filter_constructor(
    kernel_size: Tuple[int, int],
    sigma_phi: float,
    sigma_rho: float,
    theta_degrees: float,
    lambda_phi: float,
    dc_correction: bool,
) -> None:
    g_filter = gabor_filters.GaborFilter(
        kernel_size=kernel_size,
        sigma_phi=sigma_phi,
        sigma_rho=sigma_rho,
        theta_degrees=theta_degrees,
        lambda_phi=lambda_phi,
        dc_correction=dc_correction,
    )

    assert np.max(g_filter.kernel_values.real) > np.min(g_filter.kernel_values.real)
    assert np.max(g_filter.kernel_values.imag) > np.min(g_filter.kernel_values.imag)
    assert g_filter.kernel_values.shape[0] == kernel_size[1]
    assert g_filter.kernel_values.shape[1] == kernel_size[0]

    # Gabor filter values are complex numbers
    assert np.iscomplexobj(g_filter.kernel_values)

    # zero DC component
    assert math.isclose(np.mean(g_filter.kernel_values.real), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.mean(g_filter.kernel_values.imag), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.linalg.norm(g_filter.kernel_values.real, ord="fro"), 1.0, rel_tol=1e-03, abs_tol=1e-03)


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_phi, dc_correction",
    [
        (11, 2, 4, 45, 10, True),
        ((20, 21), 2, 4, 45, 10, True),
        ((4.5, 9.78), 2, 4, 45, 10, True),
        (("r", "100"), 2, 4, 45, 10, True),
        ((-1, 0), 2, 4, 45, 10, True),
        ((11, 100), 2, 4, 45, 10, True),
        ((1, 2), 2, 4, 45, 10, True),
        ((11, 15), -2, 4, 45, 10, True),
        ((15, 11), 0, 4, 45, 10, True),
        ((31, 37), 32, 1e-03, 0, 10, True),
        ((11, 15), 3, 0, 45, 10, True),
        ((15, 11), 3, -0.2, 45, 10, True),
        ((31, 21), 3, 25, 0, 10, True),
        ((31, 21), 3, 5, -5, 10, True),
        ((31, 21), 3, 5, 360, 10, True),
        ((31, 21), 3, 5, 30, 1e-03, True),
        ((31, 21), 3, 5, 30, -5, True),
        ((31, 21), 3, 5, 30, 2, "a"),
        ((31, 21), 3, 5, 30, 2, 0.1),
    ],
    ids=[
        "kernel_size is not a single number",
        "kernel_size not odd numbers",
        "kernel_size not integers1",
        "kernel_size not integers2",
        "kernel_size not positive integers",
        "kernel_size size larger than 99",
        "kernel_size size less than 3",
        "sigma_phi not positive interger1",
        "sigma_phi not positive interger2",
        "sigma_phi bigger than kernel_size[0]",
        "sigma_rho not positive interger1",
        "sigma_rho not positive interger2",
        "sigma_rho bigger than kernel_size[1]",
        "theta_degrees is not higher than/equal to 0",
        "theta_degrees is not lower than 360",
        "lambda_phi is not larger than/equal to 2",
        "lambda_phi is not positive",
        "dc_correction is not of boolean type",
        "dc_correction is not of boolean type again",
    ],
)
def test_gabor_filter_constructor_raises_an_exception(
    kernel_size: Tuple[int, int],
    sigma_phi: float,
    sigma_rho: float,
    theta_degrees: float,
    lambda_phi: float,
    dc_correction: bool,
) -> None:
    with pytest.raises((ValidationError, ImageFilterError)):
        _ = gabor_filters.GaborFilter(
            kernel_size=kernel_size,
            sigma_phi=sigma_phi,
            sigma_rho=sigma_rho,
            theta_degrees=theta_degrees,
            lambda_phi=lambda_phi,
            dc_correction=dc_correction,
        )


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_rho",
    [
        ((19, 21), np.pi / np.sqrt(2) / 2, 1, 45, 3),
        ((11, 15), 1.1, 0.5, 0, 3),
    ],
    ids=[
        "regular1",
        "regular2",
    ],
)
def test_log_gabor_filter_constructor(
    kernel_size: Tuple[int, int], sigma_phi: float, sigma_rho: float, theta_degrees: float, lambda_rho: float
) -> None:
    logg_filter = gabor_filters.LogGaborFilter(
        kernel_size=kernel_size,
        sigma_phi=sigma_phi,
        sigma_rho=sigma_rho,
        theta_degrees=theta_degrees,
        lambda_rho=lambda_rho,
    )

    assert np.max(logg_filter.kernel_values.real) > np.min(logg_filter.kernel_values.real)
    assert np.max(logg_filter.kernel_values.imag) > np.min(logg_filter.kernel_values.imag)
    assert logg_filter.kernel_values.shape[0] == kernel_size[1]
    assert logg_filter.kernel_values.shape[1] == kernel_size[0]

    # LogGabor filter values are complex numbers
    assert np.iscomplexobj(logg_filter.kernel_values)

    # zero DC component
    assert math.isclose(np.mean(logg_filter.kernel_values.real), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.mean(logg_filter.kernel_values.imag), 0.0, rel_tol=1e-03, abs_tol=1e-03)
    assert math.isclose(np.linalg.norm(logg_filter.kernel_values.real, ord="fro"), 1.0, rel_tol=1e-03, abs_tol=1e-03)


@pytest.mark.parametrize(
    "kernel_size,sigma_phi,sigma_rho,theta_degrees,lambda_rho",
    [
        (11, np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((20, 21), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((4.5, 9.78), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        (("r", "100"), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((-1, 0), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((11, 100), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((1, 2), np.pi / np.sqrt(2) / 2, 4, 45, 3),
        ((11, 15), -2, 4, 45, 10),
        ((15, 11), 0, 4, 45, 10),
        ((31, 37), 2 * np.pi, 45, 0, 10),
        ((11, 15), 0.8, 0.05, 45, 10),
        ((15, 11), 0.8, -0.2, 45, 10),
        ((31, 21), 0.8, 1.1, 0, 10),
        ((31, 21), 0.8, 0.5, -5, 10),
        ((31, 21), 0.8, 0.5, 360, 10),
        ((31, 21), 0.8, 0.5, 30, 1e-03),
        ((31, 21), 0.8, 0.5, 30, -5),
    ],
    ids=[
        "kernel_size is not a single number",
        "kernel_size not odd numbers",
        "kernel_size not integers1",
        "kernel_size not integers2",
        "kernel_size not positive integers",
        "kernel_size size larger than 99",
        "kernel_size size less than 3",
        "sigma_phi not positive interger1",
        "sigma_phi not positive interger2",
        "sigma_phi bigger than np.pi",
        "sigma_rho not positive interger1",
        "sigma_rho not positive interger2",
        "sigma_rho bigger than 1",
        "theta_degrees is not higher than/equal to 0",
        "theta_degrees is not lower than 360",
        "lambda_phi is not larger than/equal to 2",
        "lambda_phi is not positive",
    ],
)
def test_log_gabor_filter_constructor_raises_an_exception(
    kernel_size: Tuple[int, int], sigma_phi: float, sigma_rho: float, theta_degrees: float, lambda_rho: float
) -> None:
    with pytest.raises((ValidationError, ImageFilterError)):
        _ = gabor_filters.LogGaborFilter(
            kernel_size=kernel_size,
            sigma_phi=sigma_phi,
            sigma_rho=sigma_rho,
            theta_degrees=theta_degrees,
            lambda_rho=lambda_rho,
        )
