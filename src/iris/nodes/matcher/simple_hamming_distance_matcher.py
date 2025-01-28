from pydantic import confloat

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

        normalise: bool
        norm_mean: confloat(ge=0, le=1)

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: int = 15,
        normalise: bool = False,
        norm_mean: float = 0.45,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int, optional): Rotation shifts allowed in matching (in columns). Defaults to 15.
            normalise (bool, optional): Flag to normalize HD. Defaults to False.
            norm_mean (float, optional): Peak of the non-match distribution. Defaults to 0.45.
        """
        super().__init__(rotation_shift=rotation_shift, normalise=normalise, norm_mean=norm_mean)

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
        )
        return score
