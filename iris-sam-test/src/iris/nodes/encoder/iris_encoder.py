from typing import List

import numpy as np
from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisFilterResponse, IrisTemplate


class IrisEncoder(Algorithm):
    """Binarize IrisFilterResponse to generate iris code using Daugman's method.

    Algorithm steps:
        1) Binarize iris response by comparing real and imaginary parts to zero.
        2) Binarize mask response by comparing real and imaginary parts to a given parameter: mask_threshold.

    Reference:
        [1] https://www.robots.ox.ac.uk/~az/lectures/est/iris.pdf.
    """

    class Parameters(Algorithm.Parameters):
        """IrisEncoder parameters."""

        mask_threshold: float = Field(..., ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(self, mask_threshold: float = 0.9, callbacks: List[Callback] = []) -> None:
        """Assign parameters.

        Args:
            mask_threshold (float): threshold to binarize mask_responses, in the range of [0,1]. Defaults to 0.9.
            callbacks (List[Callback]): callbacks list. Defaults to [].
        """
        super().__init__(mask_threshold=mask_threshold, callbacks=callbacks)

    def run(self, response: IrisFilterResponse) -> IrisTemplate:
        """Encode iris code and mask code.

        Args:
            response (IrisFilterResponse): Filter responses.

        Returns:
            IrisTemplate: Final iris template.
        """
        iris_codes: List[np.ndarray] = []
        mask_codes: List[np.ndarray] = []

        for iris_response, mask_response in zip(response.iris_responses, response.mask_responses):
            mask_code = mask_response >= self.params.mask_threshold

            iris_code = np.stack([iris_response.real > 0, iris_response.imag > 0], axis=-1)
            mask_code = np.stack([mask_code, mask_code], axis=-1)

            iris_codes.append(iris_code)
            mask_codes.append(mask_code)

        return IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version=response.iris_code_version)
