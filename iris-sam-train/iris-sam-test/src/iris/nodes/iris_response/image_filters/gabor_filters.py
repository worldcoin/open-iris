from typing import Any, Dict, Tuple

import numpy as np
from pydantic import Field, conint, root_validator, validator

import iris.io.validators as pydantic_v
from iris.io.errors import ImageFilterError
from iris.nodes.iris_response.image_filters.image_filter_interface import ImageFilter


def upper_bound_Gabor_parameters(cls: type, values: Dict[str, Any]) -> Dict[str, Any]:
    """Check upper bounds of Gabor filter parameters such as sigma_phi, sigma_rho and lambda_phi for the given kernel_size.

    Args:
        cls (type): class type.
        values (Dict[str, Any]): values to be checked.

    Raises:
        ImageFilterError: Raised if 1) sigma_phi is greater than kernel_size[0], 2) sigma_rho is greater than kernel_size[1], 3) lambda_phi greater than kernel_size[0].

    Returns:
        Dict[str, Any]:  values of checked parameters.
    """
    kernel_size, sigma_phi, sigma_rho, lambda_phi = (
        values["kernel_size"],
        values["sigma_phi"],
        values["sigma_rho"],
        values["lambda_phi"],
    )

    if sigma_phi >= kernel_size[0]:
        raise ImageFilterError("Invalid parameters: sigma_phi can not be greater than kernel_size[0].")
    if sigma_rho >= kernel_size[1]:
        raise ImageFilterError("Invalid parameters: sigma_rho can not be greater than kernel_size[1].")
    if lambda_phi >= kernel_size[0]:
        raise ImageFilterError("Invalid parameters: lambda_phi can not be greater than kernel_size[0].")

    return values


def upper_bound_LogGabor_parameters(cls: type, values: Dict[str, Any]) -> Dict[str, Any]:
    """Check upper bound of LogGabor filter parameter lambda_rho for the given kernel_size.

    Args:
        cls (type): class type.
        values (Dict[str, Any]): values to be checked.

    Raises:
        ImageFilterError: lambda_phi can not be greater than kernel_size[1].

    Returns:
        Dict[str, Any]: values of checked parameters.
    """
    kernel_size, lambda_rho = values["kernel_size"], values["lambda_rho"]

    if lambda_rho >= kernel_size[1]:
        raise ImageFilterError("Invalid parameters: lambda_rho can not be greater than kernel_size[1].")

    return values


def get_xy_mesh(kernel_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Get (x,y) meshgrids for a given kernel size.

    Args:
        kernel_size (Tuple[int, int]): Kernel width and height.

    Returns:
        Tuple[np.ndarray, np.ndarray]: meshgrid of (x, y) positions.
    """
    ksize_phi_half = kernel_size[0] // 2
    ksize_rho_half = kernel_size[1] // 2

    y, x = np.meshgrid(
        np.arange(-ksize_phi_half, ksize_phi_half + 1),
        np.arange(-ksize_rho_half, ksize_rho_half + 1),
        indexing="xy",
        sparse=True,
    )

    return x, y


def get_radius(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Get radius to the image center for a given array of relative positions (x,y).

    Args:
        x (np.ndarray): x position relative to the image center.
        y (np.ndarray): y position relative to the image center.

    Returns:
        np.ndarray: radius to the image center.
    """
    radius = np.sqrt(x**2 + y**2)

    return radius


def rotate(x: np.ndarray, y: np.ndarray, angle: float) -> Tuple[np.ndarray, np.ndarray]:
    """Rotate a given array of relative positions (x,y) by a given angle.

    Args:
        x (np.ndarray): x position.
        y (np.ndarray): y position.
        angle (float): angle for rotation (in degrees).

    Returns:
        Tuple[np.ndarray, np.ndarray]: rotated x, y positions.
    """
    cos_theta = np.cos(angle * np.pi / 180)
    sin_theta = np.sin(angle * np.pi / 180)

    rotx = x * cos_theta + y * sin_theta
    roty = -x * sin_theta + y * cos_theta

    return rotx, roty


def normalize_kernel_values(kernel_values: np.ndarray) -> np.ndarray:
    """Normalize the kernel values so that the square sum is 1.

    Args:
        kernel_values (np.ndarray): Kernel values (complex numbers).

    Returns:
        np.ndarray: normalized Kernel values.
    """
    norm_real = np.linalg.norm(kernel_values.real, ord="fro")

    if norm_real > 0:
        kernel_values.real /= norm_real

    norm_imag = np.linalg.norm(kernel_values.imag, ord="fro")
    if norm_imag > 0:
        kernel_values.imag /= norm_imag

    return kernel_values


def convert_to_fixpoint_kernelvalues(kernel_values: np.ndarray) -> np.ndarray:
    """Convert the kernel values (both real and imaginary) to fix points.

    Args:
        kernel_values (np.ndarray): Kernel values.

    Returns:
        np.ndarray: fix-point Kernel values.
    """
    if np.iscomplexobj(kernel_values):
        kernel_values.real = np.round(kernel_values.real * 2**15)
        kernel_values.imag = np.round(kernel_values.imag * 2**15)
    else:
        kernel_values = np.round(kernel_values * 2**15)

    return kernel_values


class GaborFilter(ImageFilter):
    """Implementation of a 2D Gabor filter.

    Reference:
        [1] https://inc.ucsd.edu/mplab/75/media//gabor.pdf.
    """

    class Parameters(ImageFilter.Parameters):
        """GaborFilter parameters."""

        kernel_size: Tuple[conint(gt=3, lt=99), conint(gt=3, lt=99)]
        sigma_phi: float = Field(..., ge=1)
        sigma_rho: float = Field(..., ge=1)
        theta_degrees: float = Field(..., ge=0, lt=360)
        lambda_phi: float = Field(..., ge=2)
        dc_correction: bool
        to_fixpoints: bool

        _upper_bound = root_validator(pre=True, allow_reuse=True)(upper_bound_Gabor_parameters)
        _is_odd = validator("kernel_size", allow_reuse=True, each_item=True)(pydantic_v.is_odd)

    __parameters_type__ = Parameters

    def __init__(
        self,
        *,
        kernel_size: Tuple[conint(gt=3, lt=99), conint(gt=3, lt=99)],
        sigma_phi: float,
        sigma_rho: float,
        theta_degrees: float,
        lambda_phi: float,
        dc_correction: bool = True,
        to_fixpoints: bool = False,
    ) -> None:
        """Assign parameters.

        Args:
            kernel_size (Tuple[conint(gt=3, lt=99), conint(gt=3, lt=99)]): Kernel width and height.
            sigma_phi (float): phi standard deviation.
            sigma_rho (float): rho standard deviation.
            theta_degrees (float): orientation of kernel in degrees.
            lambda_phi (float): wavelength of the sinusoidal factor, lower value = thinner strip.
            dc_correction (bool, optional): whether to enable DC correction. Defaults to True.
            to_fixpoints (bool, optional): whether to convert kernel values to fixpoints. Defaults to False.

        """
        super().__init__(
            kernel_size=kernel_size,
            sigma_phi=sigma_phi,
            sigma_rho=sigma_rho,
            theta_degrees=theta_degrees,
            lambda_phi=lambda_phi,
            dc_correction=dc_correction,
            to_fixpoints=to_fixpoints,
        )

    def compute_kernel_values(self) -> np.ndarray:
        """Compute 2D Gabor filter kernel values.

        Returns:
            np.ndarray: Kernel values.
        """
        # convert to polar coordinates
        x, y = get_xy_mesh(self.params.kernel_size)
        rotx, roty = rotate(x, y, self.params.theta_degrees)

        # calculate carrier and envelope
        carrier = 1j * 2 * np.pi / self.params.lambda_phi * rotx
        envelope = -(rotx**2 / self.params.sigma_phi**2 + roty**2 / self.params.sigma_rho**2) / 2

        # calculate kernel values
        kernel_values = np.exp(envelope + carrier)
        kernel_values /= 2 * np.pi * self.params.sigma_phi * self.params.sigma_rho

        # apply DC correction
        if self.params.dc_correction:
            # Step 1: calculate mean value of Gabor Wavelet
            g_mean = np.mean(np.real(kernel_values), axis=-1)
            # Step 2: define gaussian offset
            correction_term_mean = np.mean(envelope, axis=-1)
            # Step 3: substract gaussian
            kernel_values = kernel_values - (g_mean / correction_term_mean)[:, np.newaxis] * envelope

        # normalize kernel values
        kernel_values = normalize_kernel_values(kernel_values)
        if self.params.to_fixpoints:
            kernel_values = convert_to_fixpoint_kernelvalues(kernel_values)

        return kernel_values


class LogGaborFilter(ImageFilter):
    """Implementation of a 2D LogGabor filter.

    Reference:
        [1] https://en.wikipedia.org/wiki/Log_Gabor_filter.
    """

    class Parameters(ImageFilter.Parameters):
        """LogGaborFilter parameters."""

        kernel_size: Tuple[conint(gt=3, lt=99), conint(gt=3, lt=99)]
        sigma_phi: float = Field(..., gt=0, le=np.pi)
        sigma_rho: float = Field(..., gt=0.1, le=1)
        theta_degrees: float = Field(..., ge=0, lt=360)
        lambda_rho: float = Field(..., gt=2)
        to_fixpoints: bool

        _upper_bound = root_validator(pre=True, allow_reuse=True)(upper_bound_LogGabor_parameters)
        _is_odd = validator("kernel_size", allow_reuse=True, each_item=True)(pydantic_v.is_odd)

    __parameters_type__ = Parameters

    def __init__(
        self,
        *,
        kernel_size: Tuple[conint(gt=3, lt=99), conint(gt=3, lt=99)],
        sigma_phi: float,
        sigma_rho: float,
        theta_degrees: float,
        lambda_rho: float,
        to_fixpoints: bool = False,
    ) -> None:
        """Assign parameters.

        Args:
            kernel_size (Tuple[conint(gt=3, lt=99), conint(gt=3, lt=99)]): Kernel width and height.
            sigma_phi (float): bandwidth in phi (frequency domain).
            sigma_rho (float): bandwidth in rho (frequency domain).
            theta_degrees (float): orientation of filter in degrees.
            lambda_rho (float): wavelength in rho.
            to_fixpoints (bool, optional): whether to convert kernel values to fixpoints. Defaults to False.
        """
        super().__init__(
            kernel_size=kernel_size,
            sigma_phi=sigma_phi,
            sigma_rho=sigma_rho,
            theta_degrees=theta_degrees,
            lambda_rho=lambda_rho,
            to_fixpoints=to_fixpoints,
        )

    def compute_kernel_values(self) -> np.ndarray:
        """Compute 2D LogGabor filter kernel values.

        Returns:
            np.ndarray: Kernel values.
        """
        # convert to polar coordinates
        x, y = get_xy_mesh(self.params.kernel_size)
        radius = get_radius(x, y)

        # remove 0 radius value in the center
        ksize_phi_half = self.params.kernel_size[0] // 2
        ksize_rho_half = self.params.kernel_size[1] // 2
        radius[ksize_rho_half][ksize_phi_half] = 1

        # get angular distance
        [rotx, roty] = rotate(x, y, self.params.theta_degrees)
        dtheta = np.arctan2(roty, rotx)

        # calculate envelope and orientation
        envelope = np.exp(
            -0.5 * np.log2(radius * self.params.lambda_rho / self.params.kernel_size[1]) ** 2 / self.params.sigma_rho**2
        )
        envelope[ksize_rho_half][ksize_phi_half] = 0
        orientation = np.exp(-0.5 * dtheta**2 / self.params.sigma_phi**2)

        # calculate kernel values
        kernel_values = envelope * orientation
        kernel_values = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(kernel_values)))

        # normalize kernel values
        kernel_values = normalize_kernel_values(kernel_values)
        if self.params.to_fixpoints:
            kernel_values = convert_to_fixpoint_kernelvalues(kernel_values)

        return kernel_values
