"""
Iris Template Combiner

This module implements a hybrid approach for combining multiple iris templates
from the same user to enhance recognition performance.

The algorithm combines:
1. Selective Bits Fusion - Using the most reliable bits
2. Majority Voting with Weight Templates - For consistent bits
3. Inconsistent Bit Analysis - To utilize the pattern of inconsistent bits

Author: Rostyslav Shevchenko
Date: May 22, 2025
"""

from typing import List, Tuple

import numpy as np
from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisTemplate, WeightedIrisTemplate


class MajorityVoteAggregation(Algorithm):
    """
    Class for combining multiple iris templates from the same user.

    The algorithm identifies consistent and inconsistent bits across templates,
    uses majority voting for consistent bits, creates a weight template to
    reflect bit reliability, and generates a combined mask representing
    consensus of valid regions.
    """

    class Parameters(Algorithm.Parameters):
        consistency_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
        mask_threshold: float = Field(default=0.01, ge=0.0, le=1.0)
        use_inconsistent_bits: bool = Field(default=True)
        inconsistent_bit_threshold: float = Field(default=0.4, ge=0.0, le=1.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        consistency_threshold: float = 0.75,
        mask_threshold: float = 0.01,
        use_inconsistent_bits: bool = True,
        inconsistent_bit_threshold: float = 0.4,
        callbacks: List[Callback] = [],
    ):
        """
        Initialize the IrisTemplateCombiner.

        Args:
            consistency_threshold (float): Threshold for considering a bit consistent across templates
            mask_threshold (float): Threshold for considering a mask bit valid in the combined template
            use_inconsistent_bits (bool): Whether to use inconsistent bits information
            inconsistent_bit_threshold (float): Threshold for identifying inconsistent bits
        """
        super().__init__(
            consistency_threshold=consistency_threshold,
            mask_threshold=mask_threshold,
            use_inconsistent_bits=use_inconsistent_bits,
            inconsistent_bit_threshold=inconsistent_bit_threshold,
            callbacks=callbacks,
        )

    def run(self, templates: List[IrisTemplate]) -> WeightedIrisTemplate:
        """
        Combine multiple iris templates from the same user.

        Args:
            templates (List[IrisTemplate]): List of IrisTemplate objects from the same user

        Returns:
            combined_template (WeightedIrisTemplate): Combined WeightedIrisTemplate
        """
        return self.combine_templates(templates)

    def combine_templates(self, templates: List[IrisTemplate]) -> WeightedIrisTemplate:
        """
        Combine multiple iris templates from the same user.

        Args:
            templates (List[IrisTemplate]): List of IrisTemplate objects from the same user

        Returns:
            combined_template (WeightedIrisTemplate): Combined WeightedIrisTemplate
        """
        if not templates:
            raise ValueError("No templates provided for combination")

        if len(templates) == 1:
            # If only one template, return it with uniform weights
            weights = [np.ones_like(code).astype(np.float32) for code in templates[0].iris_codes]
            return WeightedIrisTemplate.from_iris_template(templates[0], weights)

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

        weighted_template = WeightedIrisTemplate.from_iris_template(combined_template, weights)
        return weighted_template

    def _combine_wavelet_codes(
        self, iris_codes: List[np.ndarray], mask_codes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Combine iris codes and mask codes for a single wavelet.

        Args:
            iris_codes (List[np.ndarray]): List of iris codes for a single wavelet from multiple templates
            mask_codes (List[np.ndarray]): List of mask codes for a single wavelet from multiple templates

        Returns:
            combined_iris_code (np.ndarray): Combined iris code for this wavelet
            combined_mask_code (np.ndarray): Combined mask code for this wavelet
            weight (np.ndarray): Weight matrix reflecting bit reliability
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

        # Fraction of templates in which the bit is valid
        valid_mask_fraction = valid_mask_counts / num_templates

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
            self.params.inconsistent_bit_threshold if self.params.use_inconsistent_bits else 0,
        )

        # take into account the fraction of templates in which the bit is valid
        weight = weight * valid_mask_fraction

        return combined_iris_code, combined_mask_code, weight
