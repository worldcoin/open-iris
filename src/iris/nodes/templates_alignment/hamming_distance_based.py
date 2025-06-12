"""
Hamming Distance-Based Template Alignment

This module implements a hamming distance-based alignment algorithm that aligns
a set of iris templates to a reference template by finding the optimal rotation
offset that minimizes the hamming distance between each template and the reference.

The algorithm:
1. Uses the first template as a reference
2. For each subsequent template, finds the best rotation offset that minimizes hamming distance
3. Applies the rotation to align templates
4. Returns the aligned templates

Author: Rostyslav Shevchenko
Date: 12 June 2025
"""

from typing import List

import numpy as np
from pydantic import Field, conint

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.utils import simple_hamming_distance


class HammingDistanceBasedAlignment(Algorithm):
    """
    Hamming distance-based template alignment algorithm.

    This algorithm aligns iris templates by finding the optimal rotation offset
    that minimizes the hamming distance between each template and a reference template.
    """

    class Parameters(Algorithm.Parameters):
        """HammingDistanceBasedAlignment parameters."""

        rotation_shift: conint(ge=0, strict=True) = Field(
            default=15, description="Maximum rotation shift allowed for alignment"
        )
        use_first_as_reference: bool = Field(
            default=True,
            description="Use first template as reference, otherwise use template with minimal sum of distances",
        )
        normalise: bool = Field(default=False, description="Whether to use normalized Hamming distance for alignment")

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: int = 15,
        use_first_as_reference: bool = True,
        normalise: bool = False,
        callbacks: List[Callback] = [],
    ):
        """
        Initialize the HammingDistanceBasedAlignment.

        Args:
            rotation_shift (int): Maximum rotation shift allowed for alignment. Defaults to 15.
            use_first_as_reference (bool): Whether to use first template as reference. Defaults to True.
            normalise (bool): Whether to use normalized Hamming distance for alignment. Defaults to False.
            callbacks (List[Callback]): List of callback functions. Defaults to [].
        """
        super().__init__(
            rotation_shift=rotation_shift,
            use_first_as_reference=use_first_as_reference,
            normalise=normalise,
            callbacks=callbacks,
        )

    def run(self, templates: List[IrisTemplate]) -> List[IrisTemplate]:
        """
        Align templates using hamming distance-based alignment.

        Args:
            templates (List[IrisTemplate]): List of IrisTemplate objects to align

        Returns:
            List[IrisTemplate]: Aligned templates
        """
        if not templates:
            raise ValueError("No templates provided for alignment")

        if len(templates) == 1:
            return templates

        # Find reference template
        if self.params.use_first_as_reference:
            reference_idx = 0
        else:
            reference_idx = self._find_best_reference(templates)

        reference_template = templates[reference_idx]
        aligned_templates = []

        # Align each template to the reference
        for i, template in enumerate(templates):
            if i == reference_idx:
                # Reference template doesn't need alignment
                aligned_templates.append(template)
            else:
                # Find optimal rotation for this template
                optimal_rotation = self._find_optimal_rotation(template, reference_template)

                # Apply rotation to align the template
                aligned_template = self._rotate_template(template, optimal_rotation)
                aligned_templates.append(aligned_template)

        return aligned_templates

    def _find_best_reference(self, templates: List[IrisTemplate]) -> int:
        """
        Find the template that has the minimum sum of distances to all other templates.

        Args:
            templates (List[IrisTemplate]): List of templates

        Returns:
            int: Index of the best reference template
        """
        n_templates = len(templates)
        min_sum_distance = float("inf")
        best_reference_idx = 0

        for i in range(n_templates):
            sum_distance = 0.0
            for j in range(n_templates):
                if i != j:
                    distance, _ = simple_hamming_distance(
                        templates[i],
                        templates[j],
                        rotation_shift=self.params.rotation_shift,
                        normalise=self.params.normalise,
                    )
                    sum_distance += distance

            if sum_distance < min_sum_distance:
                min_sum_distance = sum_distance
                best_reference_idx = i

        return best_reference_idx

    def _find_optimal_rotation(self, template: IrisTemplate, reference: IrisTemplate) -> int:
        """
        Find the optimal rotation offset for a template to align with reference.

        Args:
            template (IrisTemplate): Template to align
            reference (IrisTemplate): Reference template

        Returns:
            int: Optimal rotation offset
        """
        _, optimal_rotation = simple_hamming_distance(
            template,
            reference,
            rotation_shift=self.params.rotation_shift,
            normalise=self.params.normalise,
        )

        return optimal_rotation

    def _rotate_template(self, template: IrisTemplate, rotation: int) -> IrisTemplate:
        """
        Apply rotation to a template.

        Args:
            template (IrisTemplate): Template to rotate
            rotation (int): Rotation offset to apply

        Returns:
            IrisTemplate: Rotated template
        """
        if rotation == 0:
            return template

        # Rotate iris codes and mask codes
        rotated_iris_codes = []
        rotated_mask_codes = []

        for iris_code, mask_code in zip(template.iris_codes, template.mask_codes):
            # Apply circular shift
            rotated_iris_code = np.roll(iris_code, rotation, axis=1)
            rotated_mask_code = np.roll(mask_code, rotation, axis=1)

            rotated_iris_codes.append(rotated_iris_code)
            rotated_mask_codes.append(rotated_mask_code)

        return IrisTemplate(
            iris_codes=rotated_iris_codes,
            mask_codes=rotated_mask_codes,
            iris_code_version=template.iris_code_version,
        )
