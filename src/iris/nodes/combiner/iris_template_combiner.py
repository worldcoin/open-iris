"""Iris Template Combiner.

This module implements a hybrid approach for combining multiple iris templates
from the same user to enhance recognition performance.

The algorithm combines:
1. Selective Bits Fusion - Using the most reliable bits
2. Majority Voting with Weight Templates - For consistent bits
3. Fragile Bit Analysis - To utilize the pattern of fragile bits
"""

from typing import List, Tuple

import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisTemplate


class IrisTemplateCombiner(Algorithm):
    """Class for combining multiple iris templates from the same user.

    The algorithm identifies consistent and fragile bits across templates,
    uses majority voting for consistent bits, creates a weight template to
    reflect bit reliability, and generates a combined mask representing
    consensus of valid regions.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters for template combination."""

        consistency_threshold: float = Field(..., ge=0.0, le=1.0)
        mask_threshold: float = Field(..., ge=0.0, le=1.0)
        use_fragile_bits: bool = Field(default=True)
        fragile_bit_threshold: float = Field(..., ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        consistency_threshold: float = 0.75,
        mask_threshold: float = 0.5,
        use_fragile_bits: bool = True,
        fragile_bit_threshold: float = 0.4,
    ) -> None:
        """Initialize the IrisTemplateCombiner.

        Args:
            consistency_threshold: Threshold for considering a bit consistent across templates
            mask_threshold: Threshold for considering a mask bit valid in the combined template
            use_fragile_bits: Whether to use fragile bits information
            fragile_bit_threshold: Threshold for identifying fragile bits
        """
        super().__init__(
            consistency_threshold=consistency_threshold,
            mask_threshold=mask_threshold,
            use_fragile_bits=use_fragile_bits,
            fragile_bit_threshold=fragile_bit_threshold,
        )

    def run(self, templates: List[IrisTemplate]) -> Tuple[IrisTemplate, List[np.ndarray]]:
        """Combine multiple iris templates from the same user.

        Args:
            templates: List of IrisTemplate objects from the same user

        Returns:
            Tuple containing:
                - combined_template: Combined IrisTemplate
                - weights: List of weight matrices reflecting bit reliability for each wavelet
        """
        if not templates:
            raise ValueError("No templates provided for combination")

        if len(templates) == 1:
            # If only one template, return it with uniform weights
            weights = [np.ones_like(code) for code in templates[0].iris_codes]
            return templates[0], weights

        # Check that all templates have the same structure
        self._validate_templates(templates)

        # Get the number of wavelets (filter responses)
        num_wavelets = len(templates[0].iris_codes)

        # Initialize lists for combined iris codes and mask codes
        combined_iris_codes = []
        combined_mask_codes = []
        weights = []

        # Process each wavelet separately
        for wavelet_idx in range(num_wavelets):
            # Extract iris codes and mask codes for this wavelet from all templates
            iris_codes_wavelet = [template.iris_codes[wavelet_idx] for template in templates]
            mask_codes_wavelet = [template.mask_codes[wavelet_idx] for template in templates]

            # Combine iris codes and mask codes for this wavelet
            combined_iris_code, combined_mask_code, weight = self._combine_wavelet_codes(
                iris_codes_wavelet, mask_codes_wavelet
            )

            combined_iris_codes.append(combined_iris_code)
            combined_mask_codes.append(combined_mask_code)
            weights.append(weight)

        # Create combined template
        combined_template = IrisTemplate(
            iris_codes=combined_iris_codes,
            mask_codes=combined_mask_codes,
            iris_code_version=templates[0].iris_code_version,
        )

        return combined_template, weights

    def _validate_templates(self, templates: List[IrisTemplate]) -> None:
        """Validate that all templates have the same structure.

        Args:
            templates: List of IrisTemplate objects

        Raises:
            ValueError: If templates have different structures
        """
        # Check iris code version
        if not all(t.iris_code_version == templates[0].iris_code_version for t in templates):
            raise ValueError("Templates have different iris code versions")

        # Check number of wavelets
        if not all(len(t.iris_codes) == len(templates[0].iris_codes) for t in templates):
            raise ValueError("Templates have different numbers of wavelets")

        # Check dimensions of iris codes and mask codes
        for wavelet_idx in range(len(templates[0].iris_codes)):
            shape = templates[0].iris_codes[wavelet_idx].shape
            if not all(t.iris_codes[wavelet_idx].shape == shape for t in templates):
                raise ValueError(f"Iris codes for wavelet {wavelet_idx} have different shapes")

            if not all(t.mask_codes[wavelet_idx].shape == shape for t in templates):
                raise ValueError(f"Mask codes for wavelet {wavelet_idx} have different shapes")

    def _combine_wavelet_codes(
        self, iris_codes: List[np.ndarray], mask_codes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Combine iris codes and mask codes for a single wavelet.

        Args:
            iris_codes: List of iris codes for a single wavelet from multiple templates
            mask_codes: List of mask codes for a single wavelet from multiple templates

        Returns:
            Tuple containing:
                - combined_iris_code: Combined iris code for this wavelet
                - combined_mask_code: Combined mask code for this wavelet
                - weight: Weight matrix reflecting bit reliability
        """
        num_templates = len(iris_codes)
        shape = iris_codes[0].shape

        # Initialize arrays for counting votes and valid masks
        vote_counts = np.zeros(shape, dtype=float)
        valid_mask_counts = np.zeros(shape, dtype=float)

        # Count votes and valid masks
        for i in range(num_templates):
            # Count votes for iris code bits (1s)
            vote_counts += iris_codes[i] * mask_codes[i]

            # Count valid mask bits
            valid_mask_counts += mask_codes[i]

        # Calculate the fraction of votes for each bit
        vote_fractions = vote_counts / np.maximum(valid_mask_counts, 1)

        # Calculate consistency (how far from 0.5 the vote fraction is)
        # Values close to 0 or 1 are more consistent, values close to 0.5 are less consistent
        consistency = np.abs(vote_fractions - 0.5) * 2  # Scale to [0, 1]

        # Create combined iris code using majority voting
        combined_iris_code = (vote_fractions > 0.5).astype(bool)

        # Create combined mask code
        # A bit is considered valid if enough templates have it as valid
        combined_mask_code = ((valid_mask_counts / num_templates) >= self.params.mask_threshold).astype(bool)

        # Create weight matrix based on consistency
        # More consistent bits get higher weights
        weight = np.where(
            consistency >= self.params.consistency_threshold,
            consistency,
            self.params.fragile_bit_threshold if self.params.use_fragile_bits else 0,
        )

        return combined_iris_code, combined_mask_code, weight
