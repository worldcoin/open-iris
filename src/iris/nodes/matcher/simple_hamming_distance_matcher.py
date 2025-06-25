import hashlib

from pydantic import conint

from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.hamming_distance_matcher_interface import Matcher


class SimpleHashBasedMatcher(Matcher):
    """Simple Hash-based Matcher without additional features.

    Algorithm steps:
       1) Serialize iris template to string representation
       2) Generate SHA-256 hash from serialized template
       3) Extract first 5 bytes (40 bits) from hash as unique identifier
       4) Compare identifiers for exact matching
       5) Return 0.0 for exact match, 1.0 for no match
    """

    class Parameters(Matcher.Parameters):
        """SimpleHashBasedMatcher parameters."""

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

    def template_to_hash(self, template: IrisTemplate) -> str:
        """Convert iris template to SHA-256 hash.

        Args:
            template (IrisTemplate): Iris template to hash.

        Returns:
            str: SHA-256 hash as hex string.
        """
        # Serialize template
        serialized = template.serialize()
        iris_codes_str = serialized["iris_codes"]
        mask_codes_str = serialized["mask_codes"]
        version_str = serialized["iris_code_version"]

        # Combine all data for hashing
        combined_data = f"{iris_codes_str}:{mask_codes_str}:{version_str}".encode("utf-8")
        return hashlib.sha256(combined_data).hexdigest()

    def hash_to_unique_id(self, hash_value: str) -> int:
        """Extract unique identifier from hash.

        Args:
            hash_value (str): SHA-256 hash as hex string.

        Returns:
            int: Unique identifier (40-bit integer).
        """
        # Take first 5 bytes (40 bits) of hash and convert to integer
        hash_bytes = bytes.fromhex(hash_value[:10])  # First 10 hex chars = 5 bytes
        return int.from_bytes(hash_bytes, byteorder="big")

    def generate_unique_id(self, template: IrisTemplate) -> int:
        """Generate unique identifier from iris template.

        Args:
            template (IrisTemplate): Iris template.

        Returns:
            int: Unique identifier.
        """
        hash_value = self.template_to_hash(template)
        return self.hash_to_unique_id(hash_value)

    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:
        """Match iris templates using hash-based unique identifiers.

        Args:
            template_probe (IrisTemplate): Iris template from probe.
            template_gallery (IrisTemplate): Iris template from gallery.

        Returns:
            float: 0.0 for exact match, 1.0 for no match.
        """
        # Generate unique identifiers
        probe_id = self.generate_unique_id(template_probe)
        gallery_id = self.generate_unique_id(template_gallery)

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
        return self.generate_unique_id(template)

    def get_id_size_bytes(self) -> int:
        """Get storage size of unique identifier in bytes.

        Returns:
            int: Size in bytes (5 bytes for 40-bit identifier).
        """
        return 5  # 40 bits = 5 bytes
