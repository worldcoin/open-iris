from typing import List, Tuple

import numpy as np
from pydantic import root_validator, validator

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisFilterResponse, NormalizedIris
from iris.io.validators import are_lengths_equal, iris_code_version_check, is_not_empty
from iris.nodes.iris_response.image_filters.gabor_filters import GaborFilter
from iris.nodes.iris_response.image_filters.image_filter_interface import ImageFilter
from iris.nodes.iris_response.probe_schemas.probe_schema_interface import ProbeSchema
from iris.nodes.iris_response.probe_schemas.regular_probe_schema import RegularProbeSchema


def polar_img_padding(img: np.ndarray, p_rows: int, p_cols: int) -> np.ndarray:
    """Apply zero-padding vertically and rotate-padding horizontally to a normalized image in polar coordinates.

    Args:
        img (np.ndarray): normalized image in polar coordinates.
        p_rows (int): padding size on top and bottom.
        p_cols (int): padding size on left and right.

    Returns:
        np.ndarray: padded image.
    """
    i_rows, i_cols = img.shape
    padded_image = np.zeros((i_rows + 2 * p_rows, i_cols + 2 * p_cols))

    padded_image[p_rows : i_rows + p_rows, p_cols : i_cols + p_cols] = img
    padded_image[p_rows : i_rows + p_rows, 0:p_cols] = img[:, -p_cols:]
    padded_image[p_rows : i_rows + p_rows, -p_cols:] = img[:, 0:p_cols]

    return padded_image


class ConvFilterBank(Algorithm):
    """Apply filter bank.

    Algorithm steps:
        1) Obtain filters and corresponding probe schemas.
        2) Apply convolution to a given pair of normalized iris image using the filters and probe schemas.
        3) Generate the iris response and corresponding mask response.
    """

    class Parameters(Algorithm.Parameters):
        """Default ConvFilterBank parameters."""

        filters: List[ImageFilter]
        probe_schemas: List[ProbeSchema]
        iris_code_version: str

        # Validators
        _are_lengths_equal = root_validator(pre=True, allow_reuse=True)(are_lengths_equal("probe_schemas", "filters"))
        _is_not_empty = validator("filters", "probe_schemas", allow_reuse=True)(is_not_empty)
        _iris_code_version_check = validator("iris_code_version", allow_reuse=True)(iris_code_version_check)

    __parameters_type__ = Parameters

    def __init__(
        self,
        iris_code_version: str = "v0.1",
        filters: List[ImageFilter] = [
            GaborFilter(
                kernel_size=(41, 21),
                sigma_phi=7,
                sigma_rho=6.13,
                theta_degrees=90.0,
                lambda_phi=28,
                dc_correction=True,
                to_fixpoints=True,
            ),
            GaborFilter(
                kernel_size=(17, 21),
                sigma_phi=2,
                sigma_rho=5.86,
                theta_degrees=90.0,
                lambda_phi=8,
                dc_correction=True,
                to_fixpoints=True,
            ),
        ],
        probe_schemas: List[ProbeSchema] = [
            RegularProbeSchema(n_rows=16, n_cols=256),
            RegularProbeSchema(n_rows=16, n_cols=256),
        ],
    ) -> None:
        """Assign parameters.

        Args:
            iris_code_version (str): Iris code version. Defaults to "v0.1".
            filters (List[ImageFilter]): List of image filters.
            probe_schemas (List[ProbeSchema]): List of corresponding probe schemas.
        """
        super().__init__(filters=filters, probe_schemas=probe_schemas, iris_code_version=iris_code_version)

    def run(self, normalization_output: NormalizedIris) -> IrisFilterResponse:
        """Apply filters to a normalized iris image.

        Args:
            normalization_output (NormalizedIris): Output of the normalization process.

        Returns:
            IrisFilterResponse: filter responses.
        """
        iris_responses: List[np.ndarray] = []
        mask_responses: List[np.ndarray] = []

        for i_filter, i_schema in zip(self.params.filters, self.params.probe_schemas):
            iris_response, mask_response = self._convolve(i_filter, i_schema, normalization_output)
            iris_responses.append(iris_response)
            mask_responses.append(mask_response)

        return IrisFilterResponse(
            iris_responses=iris_responses,
            mask_responses=mask_responses,
            iris_code_version=self.params.iris_code_version,
        )

    def _convolve(
        self, img_filter: ImageFilter, probe_schema: ProbeSchema, normalization_output: NormalizedIris
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply convolution to a given normalized iris image with the filter and probe schema.

        Args:
            img_filter (ImageFilter): filter used for convolution.
            probe_schema (ProbeSchema): probe schema used for convolution.
            normalization_output (NormalizedIris): Output of the normalization process.

        Returns:
            Tuple[np.ndarray, np.ndarray]: iris response and mask response.
        """
        i_rows, i_cols = normalization_output.normalized_image.shape
        k_rows, k_cols = img_filter.kernel_values.shape
        p_rows = k_rows // 2
        p_cols = k_cols // 2
        iris_response = np.zeros((probe_schema.params.n_rows, probe_schema.params.n_cols), dtype=np.complex64)
        mask_response = np.zeros((probe_schema.params.n_rows, probe_schema.params.n_cols))

        padded_iris = polar_img_padding(normalization_output.normalized_image, 0, p_cols)
        padded_mask = polar_img_padding(normalization_output.normalized_mask, 0, p_cols)

        for i in range(probe_schema.params.n_rows):
            for j in range(probe_schema.params.n_cols):
                # Convert probe_schema position to integer pixel position.
                pos = i * probe_schema.params.n_cols + j
                r_probe = min(round(probe_schema.rhos[pos] * i_rows), i_rows - 1)
                c_probe = min(round(probe_schema.phis[pos] * i_cols), i_cols - 1)

                # Get patch from image centered at [i,j] probed pixel position.
                rtop = max(0, r_probe - p_rows)
                rbot = min(r_probe + p_rows + 1, i_rows - 1)
                iris_patch = padded_iris[rtop:rbot, c_probe : c_probe + k_cols]
                mask_patch = padded_mask[rtop:rbot, c_probe : c_probe + k_cols]

                # Perform convolution at [i,j] probed pixel position.
                ktop = p_rows - iris_patch.shape[0] // 2
                iris_response[i][j] = (
                    (iris_patch * img_filter.kernel_values[ktop : ktop + iris_patch.shape[0], :]).sum()
                    / iris_patch.shape[0]
                    / k_cols
                )
                mask_response[i][j] = (
                    0 if iris_response[i][j] == 0 else (mask_patch.sum() / iris_patch.shape[0] / k_cols)
                )

        return iris_response, mask_response
