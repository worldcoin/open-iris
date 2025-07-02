"""
Hamming Distance-Based Template Alignment

This module implements a hamming distance-based alignment algorithm that aligns
iris templates to a reference template by finding the optimal rotation offset
that minimizes the hamming distance between each template and the reference.

The algorithm:
1. Finds the best reference template using original distances
2. For each template, finds the best rotation offset and aligns it to the reference
3. Returns both aligned templates and the original pairwise distances (which are invariant to global rotation)

Note: The pairwise distances are computed as the minimum Hamming distance over all possible rotations for each pair, so they are invariant to the global orientation of the templates. Thus, the distances before and after alignment are the same.

Author: Rostyslav Shevchenko
Date: 12 June 2025
"""

from enum import Enum
from typing import Dict, List

import numpy as np
from pydantic import Field, conint

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import AlignedTemplates, DistanceMatrix, IrisTemplate
from iris.nodes.matcher.utils import simple_hamming_distance


class ReferenceSelectionMethod(str, Enum):
    """Enumeration of reference selection methods."""

    LINEAR = "linear"
    MEAN_SQUARED = "mean_squared"
    ROOT_MEAN_SQUARED = "root_mean_squared"


class HammingDistanceBasedAlignment(Algorithm):
    """
    Hamming distance-based template alignment algorithm.

    This algorithm aligns iris templates by finding the optimal rotation offset
    that minimizes the hamming distance between each template and a reference template.
    Always returns both the aligned templates and their pairwise distances (computed before alignment).
    """

    class Parameters(Algorithm.Parameters):
        """HammingDistanceBasedAlignment parameters."""

        rotation_shift: conint(ge=0, strict=True) = Field(
            default=15, description="Maximum rotation shift allowed for alignment"
        )
        use_first_as_reference: bool = Field(
            default=False,
            description="Use first template as reference, otherwise use template with minimal distance aggregate",
        )
        normalise: bool = Field(default=True, description="Whether to use normalized Hamming distance for alignment")
        reference_selection_method: ReferenceSelectionMethod = Field(
            default=ReferenceSelectionMethod.LINEAR,
            description="Method for aggregating distances when selecting reference template",
        )

    __parameters_type__ = Parameters

    def __init__(
        self,
        rotation_shift: int = 15,
        use_first_as_reference: bool = False,
        normalise: bool = True,
        reference_selection_method: ReferenceSelectionMethod = ReferenceSelectionMethod.LINEAR,
        callbacks: List[Callback] = [],
    ):
        """
        Initialize the HammingDistanceBasedAlignment.

        Args:
            rotation_shift (int): Maximum rotation shift allowed for alignment. Defaults to 15.
            use_first_as_reference (bool): Whether to use first template as reference. Defaults to False.
            normalise (bool): Whether to use normalized Hamming distance for alignment. Defaults to True.
            reference_selection_method (ReferenceSelectionMethod): Method for distance aggregation when selecting reference template for alignment. Defaults to LINEAR.
            callbacks (List[Callback]): List of callback functions. Defaults to [].
        """
        super().__init__(
            rotation_shift=rotation_shift,
            use_first_as_reference=use_first_as_reference,
            normalise=normalise,
            reference_selection_method=reference_selection_method,
            callbacks=callbacks,
        )

    def run(self, templates: List[IrisTemplate]) -> AlignedTemplates:
        """
        Align templates using hamming distance-based alignment.

        Args:
            templates (List[IrisTemplate]): List of IrisTemplate objects to align

        Returns:
            AlignedTemplates: an AlignedTemplates object

        Raises:
            ValueError: If no templates provided
        """
        if not templates:
            raise ValueError("No templates provided for alignment")

        if len(templates) == 1:
            return AlignedTemplates(
                templates=templates,
                distances=DistanceMatrix(data={}),
                reference_template_id=0,
            )

        # Step 1: Calculate pairwise distances (invariant to global rotation)
        original_distances = self._calculate_pairwise_distances(templates)

        # Step 2: Find the best reference template using original distances
        if self.params.use_first_as_reference:
            reference_idx = 0
        else:
            reference_idx = self._find_best_reference(templates, original_distances)

        reference_template = templates[reference_idx]
        aligned_templates = []

        # Step 3: Align each template to the reference
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

        at = AlignedTemplates(
            templates=aligned_templates,
            distances=DistanceMatrix(data=original_distances),
            reference_template_id=reference_idx,
        )
        return at

    def _calculate_pairwise_distances(self, templates: List[IrisTemplate]) -> Dict[tuple, float]:
        """
        Calculate pairwise Hamming distances between all templates.

        Args:
            templates (List[IrisTemplate]): Templates to compare

        Returns:
            Dict[tuple, float]: Dictionary with (i, j) as keys and distances as values
        """
        distances = {}
        n_templates = len(templates)

        for i in range(n_templates):
            for j in range(i + 1, n_templates):
                distance, _ = simple_hamming_distance(
                    templates[i],
                    templates[j],
                    rotation_shift=self.params.rotation_shift,
                    normalise=self.params.normalise,
                )
                distances[(i, j)] = distance

        return distances

    def _aggregate_distances(self, distances: List[float]) -> float:
        """
        Aggregate a list of distances using the configured method.

        Args:
            distances (List[float]): List of distances to aggregate

        Returns:
            float: Aggregated distance value
        """
        if not distances:
            return 0.0

        distances_array = np.array(distances)

        if self.params.reference_selection_method == ReferenceSelectionMethod.LINEAR:
            return float(np.sum(distances_array))
        elif self.params.reference_selection_method == ReferenceSelectionMethod.MEAN_SQUARED:
            return float(np.mean(distances_array**2))
        elif self.params.reference_selection_method == ReferenceSelectionMethod.ROOT_MEAN_SQUARED:
            return float(np.sqrt(np.mean(distances_array**2)))
        else:
            raise ValueError(f"Unknown reference selection method: {self.params.reference_selection_method}")

    def _find_best_reference(self, templates: List[IrisTemplate], distances: Dict[tuple, float]) -> int:
        """
        Find the template that has the minimum aggregated distance to all other templates.

        Args:
            templates (List[IrisTemplate]): List of templates
            distances (Dict[tuple, float]): Precomputed pairwise distances

        Returns:
            int: Index of the best reference template
        """
        n_templates = len(templates)
        min_aggregated_distance = float("inf")
        best_reference_idx = 0

        for i in range(n_templates):
            template_distances = []
            for j in range(n_templates):
                if i != j:
                    # Get distance from precomputed dict
                    key = (min(i, j), max(i, j))
                    template_distances.append(distances[key])

            aggregated_distance = self._aggregate_distances(template_distances)

            if aggregated_distance < min_aggregated_distance:
                min_aggregated_distance = aggregated_distance
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
