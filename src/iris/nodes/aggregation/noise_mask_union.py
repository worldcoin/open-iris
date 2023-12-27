from typing import List

import numpy as np

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import NoiseMask


class NoiseMaskUnion(Algorithm):
    """Aggregate several NoiseMask into one by computing their union. I.E. For every bit of the NoiseMask, the output is an OR of the same bit across all NoiseMasks."""

    def run(self, elements: List[NoiseMask]) -> NoiseMask:
        """Compute the union of a list of NoiseMask.

        Args:
            elements (List[NoiseMask]): input NoiseMasks.

        Raises:
            ValueError: if not all NoiseMask.mask do not have the same shape.

        Returns:
            NoiseMask: aggregated NoiseMasks
        """
        if not all([mask.mask.shape == elements[0].mask.shape for mask in elements]):
            raise ValueError(
                f"Every NoiseMask.mask must have the same shape to be aggregated. "
                f"Received {[mask.mask.shape for mask in elements]}"
            )

        noise_union = np.sum([mask.mask for mask in elements], axis=0) > 0

        return NoiseMask(mask=noise_union)
