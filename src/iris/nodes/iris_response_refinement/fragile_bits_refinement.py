from enum import Enum
from typing import List, Tuple

import numpy as np
from pydantic import confloat

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisFilterResponse


class FragileType(str, Enum):
    """Makes wrapper for params."""

    cartesian = "cartesian"
    polar = "polar"


class FragileBitRefinement(Algorithm):
    """Calculate fragile bits for mask.

    Algorithm:
        Thresholding by the given parameter value_threshold at each bit, set the corresponding mask response to 0 if iris response is outside the thresholds.
    """

    class Parameters(Algorithm.Parameters):
        """FragileBitRefinement parameters."""

        value_threshold: Tuple[confloat(ge=0), confloat(ge=0), confloat(ge=0)]
        fragile_type: FragileType
        maskisduplicated: bool

    __parameters_type__ = Parameters

    def __init__(
        self,
        value_threshold: Tuple[confloat(ge=0), confloat(ge=0), confloat(ge=0)],
        fragile_type: FragileType = FragileType.polar,
        maskisduplicated: bool = True,
        callbacks: List[Callback] = [],
    ) -> None:
        """Create Fragile Bit Refinement object.

        Args:
            value_threshold (Tuple[confloat(ge=0), confloat(ge=0), confloat(ge=0)]): Threshold at which response
                is strong enough such that the bit is valued as 1 in the mask.
            fragile_type (FragileType, optional): The Fragile bits can be either
                calculated in cartesian or polar coordinates. In the first, the values
                of value_threshold denote to x and y axis, in the case of polar coordinates,
                the values denote to radius and angle. Defaults to FragileType.polar.
            maskisduplicated (bool, optional): If True, the mask is duplicated for both real and imaginary parts.
            callbacks(List[Callback]): List of callbacks. Defaults to [].
        """
        super().__init__(value_threshold=value_threshold, fragile_type=fragile_type, maskisduplicated=maskisduplicated, callbacks=callbacks)
        
    def run(self, response: IrisFilterResponse) -> IrisFilterResponse:
        """Generate refined IrisFilterResponse.


        Args:
            response (IrisFilterResponse): Filter bank response

        Returns:
            IrisFilterResponse: Filter bank response.
        """
        fragile_masks = []
        for iris_response, iris_mask in zip(response.iris_responses, response.mask_responses):
            if self.params.fragile_type == FragileType.cartesian:
                mask_value_imag = (
                    np.logical_and(
                        np.abs(iris_response.imag) >= self.params.value_threshold[0],
                        np.abs(iris_response.imag) <= self.params.value_threshold[2],
                    )
                    * iris_mask.imag
                )
                if self.params.maskisduplicated:
                    fragile_masks.append(mask_value_imag + 1j * mask_value_imag)
                else:
                    mask_value_real = (
                        np.logical_and(
                            np.abs(iris_response.real) >= self.params.value_threshold[0],
                            np.abs(iris_response.real) <= self.params.value_threshold[1],
                        )
                        * iris_mask.real
                    )
                    fragile_masks.append(mask_value_real + 1j * mask_value_imag)

            if self.params.fragile_type == FragileType.polar:
                # transform from cartesian to polar system
                # radius
                iris_response_r = np.abs(iris_response)
                # angle
                iris_response_phi = np.angle(iris_response)

                # min radius
                mask_value_r = np.logical_and(
                    iris_response_r >= self.params.value_threshold[0], iris_response_r <= self.params.value_threshold[1]
                )
                # min angle away from the coordinate lines
                
                if self.params.maskisduplicated:
                    mask_value = mask_value_r * iris_mask.imag
                    fragile_masks.append(mask_value + 1j * mask_value)
                else:
                    # cosine requirement: makes sure that angle is different enough from x-axis
                    cos_mask = np.abs(np.cos(iris_response_phi)) <= np.abs(np.cos(self.params.value_threshold[2]))
                    # sine requirement: makes sure that angle is different enough from y-axis
                    sine_mask = np.abs(np.sin(iris_response_phi)) <= np.abs(np.cos(self.params.value_threshold[2]))
                    # combine
                    mask_value_real = mask_value_r * sine_mask * iris_mask.real
                    # combine with radius
                    mask_value_imag = mask_value_r * cos_mask * iris_mask.imag

                    # combine with mask for response
                    fragile_masks.append(mask_value_real + 1j * mask_value_imag)

        return IrisFilterResponse(
            iris_responses=response.iris_responses,
            mask_responses=fragile_masks,
            iris_code_version=response.iris_code_version,
        )
