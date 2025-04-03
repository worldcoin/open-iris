from pydantic import confloat, conint

from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.utils import simple_hamming_distance
from iris.nodes.matcher.hamming_distance_matcher_interface import Matcher


class SimpleHammingDistanceMatcher(Matcher):
    """Hamming distance Matcher, without the bells and whistles.

    Algorithm steps:
       1) Calculate counts of nonmatch irisbits (IB_Counts) in common unmasked region and the counts of common maskbits (MB_Counts) in common unmasked region.
       2) Calculate Hamming distance (HD) based on IB_Counts and MB_Counts.
       3) If parameter `normalise` is True, normalize Hamming distance based on parameter `norm_mean` and parameter `norm_nb_bits`.
       4) If parameter rotation_shift is > 0, repeat the above steps for additional rotations of the iriscode.
       5) Return the minimium distance from above calculations.
    """

    class Parameters(Matcher.Parameters):
        """IrisMatcherParameters parameters."""

        normalise: bool
        norm_mean: confloat(ge=0, le=1)
        norm_nb_bits: conint(gt=0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: int = 15,
        normalise: bool = False,
        norm_mean: float = 0.45,
        norm_nb_bits: float = 12288,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int = 15): rotation allowed in matching, converted to columns. Defaults to 15.
            normalise (bool = False): Flag to normalize HD. Defaults to False.
            norm_mean (float = 0.45): Peak of the non-match distribution. Defaults to 0.45.
            norm_nb_bits (float = 12288): Average number of bits visible in 2 randomly sampled iris codes. Defaults to 12288 (3/4 * total_bits_number for the iris code format v0.1).

        """
        super().__init__(
            rotation_shift=rotation_shift, normalise=normalise, norm_mean=norm_mean, norm_nb_bits=norm_nb_bits
        )

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:
        """Match iris templates using Hamming distance.

        Args:
            template_probe (IrisTemplate): Iris template from probe.
            template_gallery (IrisTemplate): Iris template from gallery.

        Returns:
            float: matching distance.
        """
        score, _ = simple_hamming_distance(
            template_probe=template_probe,
            template_gallery=template_gallery,
            rotation_shift=self.params.rotation_shift,
            normalise=self.params.normalise,
            norm_mean=self.params.norm_mean,
            norm_nb_bits=self.params.norm_nb_bits,
        )
        return score
