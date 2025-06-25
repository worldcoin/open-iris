import hashlib

from pydantic import conint

from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.hamming_distance_matcher_interface import Matcher


class HashBasedMatcher(Matcher):
    """Hash-based Matcher that generates 40-bit unique identifiers.

    Algorithm steps:
       1) Use IrisTemplate's built-in generate_unique_id method
       2) Compare identifiers for exact matching
       3) Return 0.0 for exact match, 1.0 for no match
    """

    class Parameters(Matcher.Parameters):
        """HashBasedMatcher parameters."""

        rotation_shift: conint(ge=0, strict=True)  # Kept for interface compatibility
        hash_bits: int = 40  # Number of bits to extract from hash

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: conint(ge=0, strict=True) = 0,  # Not used in hash-based approach
        hash_bits: int = 40,
    ) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int): Kept for interface compatibility, not used in hash-based approach.
            hash_bits (int): Number of bits to extract from hash. Defaults to 40.
        """
        super().__init__(rotation_shift=rotation_shift)
        self.hash_bits = hash_bits

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:
        """Match iris templates using hash-based unique identifiers.

        Args:
            template_probe (IrisTemplate): Iris template from probe.
            template_gallery (IrisTemplate): Iris template from gallery.

        Returns:
            float: 0.0 for exact match, 1.0 for no match.
        """
        # Generate unique identifiers using built-in method
        probe_id = template_probe.generate_unique_id()
        gallery_id = template_gallery.generate_unique_id()

        # Compare identifiers
        if probe_id == gallery_id:
            return 0.0  # Exact match
        else:
            return 1.0  # No match

    def get_unique_id(self, template: IrisTemplate) -> int:
        """Get unique identifier for a template.

        Args:
            template (IrisTemplate): Iris template.

        Returns:
            int: Unique identifier.
        """
        return template.generate_unique_id()

    def get_id_size_bytes(self) -> int:
        """Get storage size of unique identifier in bytes.

        Returns:
            int: Size in bytes (5 bytes for 40-bit identifier).
        """
        return 5  # 40 bits = 5 bytes
