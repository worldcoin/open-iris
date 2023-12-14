from typing import List, Optional

import numpy as np
from pydantic import confloat, conint

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.utils import hamming_distance


class HammingDistanceMatcher(Algorithm):
    """Hamming distance Matcher.

    Algorithm steps:
       1) Calculate counts of nonmatch irisbits (IB_Counts) in common unmasked region and the counts of common maskbits (MB_Counts) in common unmasked region for both upper and lower half of iris, respectively.
       2) If parameter nm_dist is defined, calculate normalized Hamming distance (NHD) based on IB_Counts, MB_Counts and nm_dist.
       3) If parameter weights is defined, calculate weighted Hamming distance (WHD) based on IB_Counts, MB_Counts and weights.
       4) If parameters nm_dist and weights are both defined, calculate weighted normalized Hamming distance (WNHD) based on IB_Counts, MB_Counts, nm_dist and weights.
       5) Otherwise, calculate Hamming distance (HD) based on IB_Counts and MB_Counts.
       6) If parameter rotation_shift is > 0, repeat the above steps for additional rotations of the iriscode.
       7) Return the minimium distance from above calculations and its correpsonding rotation angle.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for HammingDistanceMatcher."""

        rotation_shift: conint(ge=0, le=180, strict=True)
        nm_dist: Optional[confloat(ge=0, le=1, strict=True)]
        weights: Optional[List[np.ndarray]]

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: int = 15,
        nm_dist: Optional[confloat(ge=0, le=1, strict=True)] = None,
        weights: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int): rotations allowed in matching, converted to shifts in columns. Defaults to 15.
            nm_dist (Optional[confloat(ge=0, le = 1, strict=True)]): nonmatch distance used for normalized HD. Optional paremeter for normalized HD. Defaults to None.
            weights (Optional[List[np.ndarray]]): list of weights table. Optional paremeter for weighted HD. Defaults to None.
        """
        super().__init__(rotation_shift=rotation_shift, nm_dist=nm_dist, weights=weights)

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:
        """Match iris templates using Hamming distance.

        Args:
            template_probe (IrisTemplate): Iris template from probe.
            template_gallery (IrisTemplate): Iris template from gallery.

        Returns:
            float: matching distance.
        """
        score, _ = hamming_distance(
            template_probe,
            template_gallery,
            self.params.rotation_shift,
            self.params.nm_dist,
            self.params.weights,
        )

        return score
