from typing import List, Optional

import numpy as np
from pydantic import confloat, conint

from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.hamming_distance_matcher_interface import Matcher
from iris.nodes.matcher.utils import hamming_distance


class HammingDistanceMatcher(Matcher):
    """Hamming distance Matcher with additional optional features.

    Algorithm steps:
       1) Calculate counts of nonmatch irisbits (IB_Counts) in common unmasked region and the counts of common maskbits (MB_Counts) in common unmasked region for both upper and lower half of iris, respectively.
       2) If parameter norm_mean is defined, calculate normalized Hamming distance (NHD) based on IB_Counts, MB_Counts and norm_mean.
       3) If parameter weights is defined, calculate weighted Hamming distance (WHD) based on IB_Counts, MB_Counts and weights.
       4) If parameters norm_mean and weights are both defined, calculate weighted normalized Hamming distance (WNHD) based on IB_Counts, MB_Counts, norm_mean and weights.
       5) Otherwise, calculate Hamming distance (HD) based on IB_Counts and MB_Counts.
       6) If parameter rotation_shift is > 0, repeat the above steps for additional rotations of the iriscode.
       7) Return the minimium distance from above calculations.
    """

    class Parameters(Matcher.Parameters):
        """HammingDistanceMatcher parameters."""

        rotation_shift: conint(ge=0, strict=True)
        normalise: bool
        norm_mean: confloat(ge=0, le=1, strict=True)
        norm_gradient: float
        separate_half_matching: bool
        weights: Optional[List[np.ndarray]]

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: conint(ge=0, strict=True) = 15,
        normalise: bool = True,
        norm_mean: confloat(ge=0, le=1, strict=True) = 0.45,
        norm_gradient: float = 0.00005,
        separate_half_matching: bool = True,
        weights: Optional[List[np.ndarray]] = None,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (Optional[conint(ge=0, strict=True)], optional): Rotation shifts allowed in matching (in columns). Defaults to 15.
            normalise (bool, optional): Flag to normalize HD. Defaults to True.
            norm_mean (Optional[confloat(ge=0, le = 1, strict=True)], optional): Nonmatch distance used for normalized HD. Optional paremeter for normalized HD. Defaults to 0.45.
            norm_gradient: float, optional): Gradient for linear approximation of normalization term. Defaults to 0.00005.
            separate_half_matching (bool, optional): Separate the upper and lower halves for matching. Defaults to True.
            weights (Optional[List[np.ndarray]], optional): list of weights table. Optional paremeter for weighted HD. Defaults to None.
        """
        super().__init__(
            rotation_shift=rotation_shift,
            normalise=normalise,
            norm_mean=norm_mean,
            norm_gradient=norm_gradient,
            separate_half_matching=separate_half_matching,
            weights=weights,
        )

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:
        """Match iris templates using Hamming distance.

        Args:
            template_probe (IrisTemplate): Iris template from probe.
            template_gallery (IrisTemplate): Iris template from gallery.

        Returns:
            float: matching distance.
        """
        score, _ = hamming_distance(
            template_probe=template_probe,
            template_gallery=template_gallery,
            rotation_shift=self.params.rotation_shift,
            normalise=self.params.normalise,
            norm_mean=self.params.norm_mean,
            norm_gradient=self.params.norm_gradient,
            separate_half_matching=self.params.separate_half_matching,
            weights=self.params.weights,
        )

        return score
