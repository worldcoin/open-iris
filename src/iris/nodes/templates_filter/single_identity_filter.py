import logging
from collections import defaultdict, deque
from enum import Enum
from typing import Dict, List, Set, Tuple

import numpy as np
from pydantic import Field, conint

import iris.io.errors as E
from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import AlignedTemplates, DistanceMatrix, IrisTemplate
from iris.nodes.matcher.utils import simple_hamming_distance


def greedy_purification(
    distances: Dict[Tuple[int, int], float], threshold: float, nb_templates: int, min_templates: int = 1
) -> List[int]:
    """
    Iteratively remove the template with the highest mean distance to others until
    all pairwise distances are within threshold or until min_templates remain.

    Args:
        distances (Dict[Tuple[int, int], float]): dict of (i, j): distance
        threshold (float): distance threshold
        nb_templates (int): number of templates
        min_templates (int): minimum templates to retain. Default is 1.

    Returns:
        List[int]: indices of outlier templates to remove
    """
    remaining = set(range(nb_templates))
    removed = set()

    # Convert dict to a matrix for convenience
    dist_matrix = np.full((nb_templates, nb_templates), np.nan)
    for (i, j), d in distances.items():
        dist_matrix[i, j] = d
        dist_matrix[j, i] = d

    max_removals = nb_templates - min_templates
    for _ in range(max_removals + 1):
        # Check if all pairwise distances among remaining are within threshold
        sub = list(remaining)
        pairs = [(i, j) for idx, i in enumerate(sub) for j in sub[idx + 1 :]]
        if all(dist_matrix[i, j] <= threshold for i, j in pairs):
            break
        if len(remaining) <= min_templates:
            break
        # Compute mean distance for each remaining template
        mean_distances = {}
        for i in remaining:
            others = [j for j in remaining if j != i]
            mean = np.nanmean([dist_matrix[i, j] for j in others])
            mean_distances[i] = mean
        # Find template with largest mean distance
        worst_score = max(mean_distances.values())
        worst_candidates = [i for i, v in mean_distances.items() if v == worst_score]
        worst = min(worst_candidates)  # Deterministic: pick the lowest index

        removed.add(worst)
        remaining.remove(worst)
    return sorted(list(removed))


def find_identity_clusters(
    distances: Dict[Tuple[int, int], float], nb_templates: int, threshold: float, min_cluster_size: int = 2
) -> List[Set[int]]:
    """
    Identifies clusters (potential identities) among templates based on pairwise distances.

    Args:
        distances (Dict[Tuple[int, int], float]): Dictionary of pairwise distances between templates, keys are (i, j) tuples.
        nb_templates (int): Total number of templates (nodes).
        threshold (float): Maximum distance to consider two templates as belonging to the same identity (edge in the graph).
        min_cluster_size (int): Minimum number of templates required to consider a group a cluster (default: 2).

    Returns:
        List[Set[int]]: List of clusters, each cluster is a set of template indices. Only clusters with size >= min_cluster_size are returned.
    """
    # Build adjacency list: connect nodes (templates) if their distance is within the threshold
    adj: Dict[int, Set[int]] = defaultdict(set)
    for (i, j), d in distances.items():
        if d <= threshold:
            adj[i].add(j)
            adj[j].add(i)

    # Find connected components (clusters) using BFS
    visited: Set[int] = set()
    clusters: List[Set[int]] = []
    for node in range(nb_templates):
        if node not in visited:
            queue = deque([node])
            cluster: Set[int] = set()
            while queue:
                current_node = queue.popleft()
                if current_node not in visited:
                    visited.add(current_node)
                    cluster.add(current_node)
                    # Add all unvisited neighbors to the queue
                    queue.extend(adj[current_node] - visited)
            # Only consider clusters of sufficient size
            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)
    return clusters


class IdentityValidationAction(str, Enum):
    REMOVE = "remove"
    RAISE_ERROR = "raise_error"
    LOG_WARNING = "log_warning"


class TemplateIdentityFilter(Algorithm):
    """
    Node for filtering iris templates to ensure all templates are likely from the same identity.
    Uses pairwise Hamming distances to detect and remove (or flag) outlier templates.
    Can operate on precomputed distances (from alignment node) or compute them internally.
    """

    class Parameters(Algorithm.Parameters):
        identity_distance_threshold: float = Field(
            default=0.35,
            ge=0.0,
            le=1.0,
            description="Maximum allowed Hamming distance between templates to consider them from the same identity.",
        )
        identity_validation_action: IdentityValidationAction = Field(
            default=IdentityValidationAction.REMOVE,
            description=f"Action if identity validation fails: {', '.join([e.value for e in IdentityValidationAction])}.",
        )
        min_templates_after_validation: conint(ge=1) = Field(
            default=1, description="Minimum number of templates required after filtering. Raise error if fewer remain."
        )

    __parameters_type__ = Parameters

    def __init__(
        self,
        identity_distance_threshold: float = 0.35,
        identity_validation_action: IdentityValidationAction = IdentityValidationAction.REMOVE,
        min_templates_after_validation: int = 1,
        callbacks: List[Callback] = [],
    ):
        """
        Initialize the TemplateIdentityFilter node.

        Args:
            identity_distance_threshold (float): Maximum allowed Hamming distance for identity validation.
            identity_validation_action (IdentityValidationAction): Action to take if validation fails.
            min_templates_after_validation (int): Minimum number of templates required after filtering.
            callbacks (List[Callback]): List of callback functions.
        """
        super().__init__(
            identity_distance_threshold=identity_distance_threshold,
            identity_validation_action=identity_validation_action,
            min_templates_after_validation=min_templates_after_validation,
            callbacks=callbacks,
        )

    def run(
        self,
        aligned_templates: AlignedTemplates,
    ) -> List[IrisTemplate]:
        """
        Filter templates to ensure all are from the same identity based on Hamming distances.

        Args:
            templates (List[IrisTemplate]): List of iris templates to validate.
            pairwise_distances (Dict[tuple, float], optional): Precomputed pairwise Hamming distances. If not provided, will be computed.

        Returns:
            List[IrisTemplate]: Filtered list of templates passing identity validation.

        Raises:
            E.IdentityValidationError: If validation fails and action is RAISE_ERROR or not enough templates remain.
        """
        if len(aligned_templates) == 1:
            return aligned_templates.templates

        templates = aligned_templates.templates
        pairwise_distances = aligned_templates.distances

        # Validate that the number of pairwise distances matches the expected count for n templates
        expected_num_distances = len(templates) * (len(templates) - 1) / 2
        if len(pairwise_distances) != expected_num_distances:
            raise E.IdentityValidationError(
                f"Number of pairwise distances is incorrect. Expected: {expected_num_distances}, "
                f"got: {len(pairwise_distances)}"
            )

        # Use provided distances or compute them if not given
        outlier_indices = self._find_identity_outliers(pairwise_distances)
        return self._handle_identity_outliers(templates, outlier_indices, pairwise_distances.data)

    def _calculate_pairwise_distances(self, templates: List[IrisTemplate]) -> DistanceMatrix:
        """
        Compute pairwise Hamming distances between all templates.

        Args:
            templates (List[IrisTemplate]): List of iris templates.

        Returns:
            Dict[tuple, float]: Dictionary with (i, j) as keys and Hamming distances as values.
        """
        distances = {}
        n = len(templates)
        for i in range(n):
            for j in range(i + 1, n):
                d, _ = simple_hamming_distance(templates[i], templates[j])
                distances[(i, j)] = d
        distances_matrix = DistanceMatrix(data=distances)
        return distances_matrix

    def _find_identity_outliers(self, distances: DistanceMatrix) -> List[int]:
        """
        Identify indices of templates that are outliers based on the identity distance threshold.

        Args:
            distances (DistanceMatrix): Pairwise Hamming distances.

        Returns:
            List[int]: Indices of templates considered outliers.
        """
        n = distances.nb_templates
        threshold = self.params.identity_distance_threshold

        # Step 1: Find clusters
        clusters = find_identity_clusters(distances.data, n, threshold, min_cluster_size=2)

        if len(clusters) == 0:
            raise E.IdentityValidationError(
                f"Found no identity clusters. All templates have Hamming distance exceeding threshold {threshold:.4f}. "
            )

        if len(clusters) > 1:
            raise E.IdentityValidationError(
                f"Found multiple identity clusters with Hamming distance exceeding threshold {threshold:.4f}. "
                f"Template indices of the clusters: {[sorted(list(c)) for c in clusters]}"
            )

        # Step 2: if there is only one cluster, treat other nodes as outliers
        outliers = [i for i in range(n) if i not in clusters[0]]

        return outliers

    def _handle_identity_outliers(
        self,
        templates: List[IrisTemplate],
        outlier_indices: List[int],
        distances: Dict[tuple, float],
    ) -> List[IrisTemplate]:
        action = self.params.identity_validation_action
        threshold = self.params.identity_distance_threshold

        if action == IdentityValidationAction.RAISE_ERROR:
            max_distance = max(distances.values())
            raise E.IdentityValidationError(
                f"Identity validation failed: Found templates with Hamming distance {max_distance:.4f} "
                f"exceeding threshold {threshold:.4f}. Template indices with violations: {outlier_indices}"
            )
        elif action == IdentityValidationAction.REMOVE:
            valid_templates = [t for i, t in enumerate(templates) if i not in outlier_indices]
            if len(valid_templates) < self.params.min_templates_after_validation:
                raise E.IdentityValidationError(
                    f"Identity validation removed too many templates. "
                    f"Remaining: {len(valid_templates)}, required: {self.params.min_templates_after_validation}"
                )
            if outlier_indices:
                logging.warning(
                    f"Identity validation: templates at indices {outlier_indices} exceed threshold {threshold}"
                )
            return valid_templates
        elif action == IdentityValidationAction.LOG_WARNING:
            if outlier_indices:
                logging.warning(
                    f"Identity validation: templates at indices {outlier_indices} exceed threshold {threshold}"
                )
            return templates
        else:
            raise ValueError(f"Unknown identity_validation_action: {action}")
