from pydantic import confloat, conint

from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.hamming_distance_matcher_interface import Matcher
from iris.nodes.matcher.utils import simple_hamming_distance


class SimpleHammingDistanceMatcher(Matcher):
    """Simple Hamming distance Matcher without the bells and whistles.

    Algorithm steps:
       1) Calculate counts of nonmatch irisbits (IB_Counts) in common unmasked region and the counts of common maskbits (MB_Counts) in common unmasked region.
       2) Calculate Hamming distance (HD) based on IB_Counts and MB_Counts.
       3) If parameter `normalise` is True, normalize Hamming distance based on parameter `norm_mean`.
       4) If parameter rotation_shift is > 0, repeat the above steps for additional rotations of the iriscode.
       5) Return the minimium distance from above calculations.
    """

    class Parameters(Matcher.Parameters):
        """SimpleHammingDistanceMatcher parameters."""

        rotation_shift: conint(ge=0, strict=True)
        normalise: bool
        norm_mean: confloat(ge=0, le=1, strict=True)
        norm_gradient: float

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: conint(ge=0, strict=True) = 15,
        normalise: bool = False,
        norm_mean: confloat(ge=0, le=1, strict=True) = 0.45,
        norm_gradient: float = 0.00005,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (Optional[conint(ge=0, strict=True)], optional): Rotation shifts allowed in matching (in columns). Defaults to 15.
            normalise (bool, optional): Flag to normalize HD. Defaults to False.
            norm_mean (Optional[confloat(ge=0, le = 1, strict=True)], optional): Nonmatch distance used for normalized HD. Optional paremeter for normalized HD. Defaults to 0.45.
            norm_gradient: float, optional): Gradient for linear approximation of normalization term. Defaults to 0.00005.
        """
        super().__init__(
            rotation_shift=rotation_shift,
            normalise=normalise,
            norm_mean=norm_mean,
            norm_gradient=norm_gradient,
        )

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:
        """Match iris templates using Hamming distance.

        Args:
            template_probe (IrisTemplate): Iris template from probe.
            template_gallery (IrisTemplate): Iris template from gallery.

        Returns:
            float: Matching distance.
        """
        score, _ = simple_hamming_distance(
            template_probe=template_probe,
            template_gallery=template_gallery,
            rotation_shift=self.params.rotation_shift,
            normalise=self.params.normalise,
            norm_mean=self.params.norm_mean,
            norm_gradient=self.params.norm_gradient,
        )
        return score
