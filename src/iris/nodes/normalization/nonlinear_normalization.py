from enum import Enum
from typing import Collection

import numpy as np
from pydantic import PositiveInt

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeOrientation, GeometryPolygons, IRImage, NoiseMask, NormalizedIris
from iris.io.errors import NormalizationError
from iris.nodes.normalization.utils import correct_orientation, generate_iris_mask, normalize_all, to_uint8


class NonlinearType(str, Enum):
    """Makes wrapper for params."""

    default = "default"
    wyatt = "wyatt"


class NonlinearNormalization(Algorithm):
    """Implementation of a normalization algorithm which uses nonlinear squared transformation to map image pixels.

    Algorithm steps:
        1) Create nonlinear grids of sampling radii based on parameters: res_in_r, intermediate_radiuses.
        2) Compute the mapping between the normalized image pixel location and the original image location.
        3) Obtain pixel values of normalized image using bilinear intepolation.


    References:
        [1] H J Wyatt, A 'minimum-wear-and-tear' meshwork for the iris, https://core.ac.uk/download/pdf/82071136.pdf
        [2] W-S Chen, J-C Li, Fast Non-linear Normalization Algorithm for Iris Recognition, https://www.scitepress.org/papers/2010/28409/28409.pdf
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for NonlinearNormalization."""

        res_in_r: PositiveInt
        intermediate_radiuses: Collection[float]
        oversat_threshold: PositiveInt

    __parameters_type__ = Parameters

    def __init__(
        self,
        res_in_r: PositiveInt = 128,
        oversat_threshold: PositiveInt = 254,
        method: NonlinearType = NonlinearType.default,
    ) -> None:
        """Assign parameters.

        Args:
            res_in_r (PositiveInt): Normalized image r resolution. Defaults to 128.
            oversat_threshold (PositiveInt, optional): threshold for masking over-satuated pixels. Defaults to 254.
            method (NonlinearType, optional): method for generating nonlinear grids. Defaults to NonlinearType.default.
        """
        super().__init__(
            res_in_r=res_in_r,
            intermediate_radiuses=np.array(
                [self._getgrids(max(0, res_in_r), p2i_ratio, method) for p2i_ratio in np.arange(0, 101)]
            ),
            oversat_threshold=oversat_threshold,
        )

    def run(
        self,
        image: IRImage,
        noise_mask: NoiseMask,
        extrapolated_contours: GeometryPolygons,
        eye_orientation: EyeOrientation,
    ) -> NormalizedIris:
        """Normalize iris using nonlinear transformation when sampling points from cartisian to polar coordinates.

        Args:
            image (IRImage): Input image to normalize.
            noise_mask (NoiseMask): Noise mask.
            extrapolated_contours (GeometryPolygons): Extrapolated contours.
            eye_orientation (EyeOrientation): Eye orientation angle.

        Returns:
            NormalizedIris: NormalizedIris object containing normalized image and iris mask.
        """
        if len(extrapolated_contours.pupil_array) != len(extrapolated_contours.iris_array):
            raise NormalizationError("The number of extrapolated iris and pupil points must be the same.")

        pupil_points, iris_points = correct_orientation(
            extrapolated_contours.pupil_array,
            extrapolated_contours.iris_array,
            eye_orientation.angle,
        )
        p2i_ratio = extrapolated_contours.pupil_diameter / extrapolated_contours.iris_diameter
        if p2i_ratio <= 0 or p2i_ratio >= 1:
            raise NormalizationError(f"Invalid pupil to iris ratio, not in the range (0,1): {p2i_ratio}.")

        iris_mask = generate_iris_mask(extrapolated_contours, noise_mask.mask)
        iris_mask[image.img_data >= self.params.oversat_threshold] = False
        src_points = self._generate_correspondences(pupil_points, iris_points, p2i_ratio)

        normalized_image, normalized_mask = normalize_all(
            image=image.img_data, iris_mask=iris_mask, src_points=src_points
        )
        normalized_iris = NormalizedIris(
            normalized_image=to_uint8(normalized_image),
            normalized_mask=normalized_mask,
        )
        return normalized_iris

    def _getgrids(self, res_in_r: PositiveInt, P_r100: PositiveInt, method: NonlinearType) -> np.ndarray:
        """Generate radius grids for nonlinear normalization based on p2i_ratio (pupil_to_iris ratio).

        Args:
            res_in_r (PositiveInt): Normalized image r resolution.
            P_r100 (PositiveInt): pupil_to_iris ratio in percentage, range in [0,100]
            method (NonlinearType): method for generating nonlinear grids.
        Returns:
            np.ndarray: nonlinear sampling grids for normalization
        """
        if method == NonlinearType.default:
            p = np.arange(28, max(74 - P_r100, P_r100 - 14)) ** 2
            q = (p - p[0]) / (p[-1] - p[0])  # Normalize q
            lin_space = np.linspace(0, 1.0, res_in_r + 1)
            grids = np.interp(lin_space, np.linspace(0, 1.0, len(q)), q)
            return grids[:-1] + np.diff(grids) / 2  # Return midpoint values

        if method == NonlinearType.wyatt:
            P_ref = 0.9
            c = 1.08
            P_r = P_r100 / 100
            n = res_in_r + 1

            i_values = np.arange(1, n)
            cos_theta_r = (i_values**c * (2 * (n - 1) ** c * P_ref + i_values**c * (1 - P_ref))) / (
                (n - 1) ** c * (1 + P_ref) * ((n - 1) ** c * P_ref + i_values**c * (1 - P_ref))
            )

            coefficients = np.vstack(
                [
                    np.ones_like(i_values) * (1 - P_r),
                    2 * P_r - (1 - P_r**2) * cos_theta_r,
                    -(1 + P_r) * P_r * cos_theta_r,
                ]
            ).T

            roots = np.array([np.roots(coeff) for coeff in coefficients])
            delta = np.array([r[r > 0][0] if np.any(r > 0) else 0 for r in roots])  # Take only the positive root
            return delta

    def _generate_correspondences(
        self, pupil_points: np.ndarray, iris_points: np.ndarray, p2i_ratio: float
    ) -> np.ndarray:
        """Generate corresponding positions in original image.

        Args:
            pupil_points (np.ndarray): Pupil bounding points. NumPy array of shape (num_points x 2).
            iris_points (np.ndarray): Iris bounding points. NumPy array of shape (num_points x 2).
            p2i_ratio (float): Pupil to iris ratio.
        Returns:
            np.ndarray: generated corresponding points.
        """

        src_points = np.array(
            [
                pupil_points + x * (iris_points - pupil_points)
                for x in self.params.intermediate_radiuses[round(100 * p2i_ratio)]
            ]
        )

        return np.round(src_points).astype(int)
