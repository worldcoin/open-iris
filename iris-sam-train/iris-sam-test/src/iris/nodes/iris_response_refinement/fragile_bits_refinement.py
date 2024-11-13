from typing import Literal, Tuple

import numpy as np
from pydantic import confloat

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisFilterResponse


class FragileBitRefinement(Algorithm):
    """Refining mask by masking out fragile bits.

    Algorithm:
        Thresholding by the given parameter value_threshold at each bit, set the corresponding mask response to 0 if iris response is below the threshold.
    """

    class Parameters(Algorithm.Parameters):
        """RegularProbeSchema parameters."""

        value_threshold: Tuple[confloat(ge=0), confloat(ge=0)]
        fragile_type: Literal["cartesian", "polar"]

    __parameters_type__ = Parameters

    def __init__(
        self,
        value_threshold: Tuple[confloat(ge=0), confloat(ge=0)],
        fragile_type: Literal["cartesian", "polar"] = "polar",
    ) -> None:
        """Create Fragile Bit Refinement object.

        Args:
            value_threshold (Tuple[confloat(ge=0), confloat(ge=0)]): Thresholding iris response values.
            fragile_type (Literal["cartesian", "polar"], optional): The Fragile bits can be either
                calculated in cartesian or polar coordinates. In the first, the values
                of value_threshold denote to x and y axis, in the case of polar coordinates,
                the values denote to radius and angle. Defaults to "polar".
        """
        super().__init__(value_threshold=value_threshold, fragile_type=fragile_type)

    def run(self, iris_filter_response: IrisFilterResponse) -> IrisFilterResponse:
        """Generate refined IrisFilterResponse.

        Args:
            iris_filter_response (IrisFilterResponse): Filter bank response.

        Returns:
            IrisFilterResponse: Filter bank response.
        """
        fragile_masks = []
        for iris_response, iris_mask in zip(iris_filter_response.iris_responses, iris_filter_response.mask_responses):
            if self.params.fragile_type == "cartesian":
                mask_value_real = np.abs(np.real(iris_response)) >= self.params.value_threshold[0]
                mask_value_imaginary = np.abs(np.imag(iris_response)) >= self.params.value_threshold[1]
                mask_value = mask_value_real * mask_value_imaginary

            if self.params.fragile_type == "polar":
                iris_response_r = np.abs(iris_response)
                iris_response_phi = np.angle(iris_response)

                mask_value_r = iris_response_r >= self.params.value_threshold[0]

                cos_mask = np.abs(np.cos(iris_response_phi)) <= np.abs(np.cos(self.params.value_threshold[1]))
                sine_mask = np.abs(np.sin(iris_response_phi)) <= np.abs(np.cos(self.params.value_threshold[1]))
                mask_value_phi = cos_mask * sine_mask
                mask_value = mask_value_r * mask_value_phi

            mask_value = mask_value * iris_mask
            fragile_masks.append(mask_value)

        return IrisFilterResponse(
            iris_responses=iris_filter_response.iris_responses,
            mask_responses=fragile_masks,
            iris_code_version=iris_filter_response.iris_code_version,
        )
