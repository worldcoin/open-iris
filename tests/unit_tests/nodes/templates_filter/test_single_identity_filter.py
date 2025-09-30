import logging

import numpy as np
import pytest

import iris.io.errors as E
from iris.io.dataclasses import AlignedTemplates, DistanceMatrix, IrisTemplate, IrisTemplateWithId
from iris.nodes.templates_filter.single_identity_filter import (
    IdentityValidationAction,
    TemplateIdentityFilter,
    find_identity_clusters,
    greedy_purification,
)


# ---- Helper functions ----
def make_dist_dict(matrix):
    """Helper to convert upper triangle of a matrix to dict[(i,j)] = value."""
    n = matrix.shape[0]
    d = {}
    for i in range(n):
        for j in range(i + 1, n):
            d[(i, j)] = matrix[i, j]
    return d


# ---- Tests for greedy_purification ----
def test_greedy_all_within_threshold():
    mat = np.array([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]])
    d = make_dist_dict(mat)
    assert greedy_purification(d, threshold=0.2, nb_templates=3) == []


def test_greedy_one_outlier():
    mat = np.array([[0, 0.1, 0.5], [0.1, 0, 0.5], [0.5, 0.5, 0]])
    d = make_dist_dict(mat)
    assert greedy_purification(d, threshold=0.2, nb_templates=3) == [2]


def test_greedy_multiple_outliers():
    mat = np.array([[0, 0.6, 0.6, 0.1], [0.6, 0, 0.6, 0.1], [0.6, 0.6, 0, 0.1], [0.1, 0.1, 0.1, 0]])
    d = make_dist_dict(mat)
    removed = greedy_purification(d, threshold=0.2, nb_templates=4)
    assert set(removed) == {0, 1}


def test_greedy_min_templates_respected():
    mat = np.array([[0, 0.9, 0.9], [0.9, 0, 0.9], [0.9, 0.9, 0]])
    d = make_dist_dict(mat)
    assert greedy_purification(d, threshold=0.2, nb_templates=3, min_templates=1) == [0, 1]
    assert greedy_purification(d, threshold=0.2, nb_templates=3, min_templates=2) == [0]


def test_greedy_single_template():
    d = {}
    assert greedy_purification(d, threshold=0.2, nb_templates=1) == []


def test_greedy_empty_input():
    assert greedy_purification({}, threshold=0.2, nb_templates=0) == []


def test_greedy_tie_breaking():
    mat = np.array([[0, 0.5, 0.5], [0.5, 0, 0.1], [0.5, 0.1, 0]])
    d = make_dist_dict(mat)
    removed = greedy_purification(d, threshold=0.2, nb_templates=3)
    assert removed == [0] or removed == [1]


def test_greedy_two_clusters():
    # 0,1 are close; 2,3 are close; between clusters is large
    mat = np.array([[0, 0.05, 0.8, 0.8], [0.05, 0, 0.8, 0.8], [0.8, 0.8, 0, 0.05], [0.8, 0.8, 0.05, 0]])
    d = make_dist_dict(mat)
    removed = greedy_purification(d, threshold=0.2, nb_templates=4)
    # Only one cluster should remain, so either [0,1] or [2,3] should be removed
    assert set(removed) == {0, 1} or set(removed) == {2, 3}


# ---- Tests for find_identity_clusters ----
def test_find_identity_clusters_all_one_cluster():
    mat = np.array([[0, 0.1, 0.1], [0.1, 0, 0.1], [0.1, 0.1, 0]])
    d = make_dist_dict(mat)
    clusters = find_identity_clusters(d, nb_templates=3, threshold=0.2)
    assert len(clusters) == 1
    assert clusters[0] == {0, 1, 2}


def test_find_identity_clusters_all_outliers():
    mat = np.array([[0, 0.5, 0.6], [0.5, 0, 0.7], [0.6, 0.7, 0]])
    d = make_dist_dict(mat)
    clusters = find_identity_clusters(d, nb_templates=3, threshold=0.2)
    assert clusters == []


def test_find_identity_clusters_two_clusters():
    mat = np.array([[0, 0.05, 0.8, 0.8], [0.05, 0, 0.8, 0.8], [0.8, 0.8, 0, 0.05], [0.8, 0.8, 0.05, 0]])
    d = make_dist_dict(mat)
    clusters = find_identity_clusters(d, nb_templates=4, threshold=0.2)
    assert len(clusters) == 2
    assert {0, 1} in clusters and {2, 3} in clusters


def test_find_identity_clusters_clusters_with_outliers():
    mat = np.array(
        [
            [0, 0.05, 0.8, 0.8, 0.8],
            [0.05, 0, 0.8, 0.8, 0.8],
            [0.8, 0.8, 0, 0.05, 0.8],
            [0.8, 0.8, 0.05, 0, 0.8],
            [0.8, 0.8, 0.8, 0.8, 0],
        ]
    )
    d = make_dist_dict(mat)
    clusters = find_identity_clusters(d, nb_templates=5, threshold=0.2)
    assert len(clusters) == 2
    assert {0, 1} in clusters and {2, 3} in clusters
    assert {4} not in clusters


def test_find_identity_clusters_empty_input():
    clusters = find_identity_clusters({}, nb_templates=0, threshold=0.2)
    assert clusters == []


def test_find_identity_clusters_min_cluster_size():
    mat = np.array([[0, 0.05, 0.8], [0.05, 0, 0.8], [0.8, 0.8, 0]])
    d = make_dist_dict(mat)
    # With min_cluster_size=2, only {0,1} is a cluster
    clusters = find_identity_clusters(d, nb_templates=3, threshold=0.2, min_cluster_size=2)
    assert clusters == [{0, 1}]
    # With min_cluster_size=3, no clusters
    clusters = find_identity_clusters(d, nb_templates=3, threshold=0.2, min_cluster_size=3)
    assert clusters == []


def test_find_identity_clusters_min_cluster_size_gt_n():
    # min_cluster_size > n should return no clusters
    mat = np.array([[0, 0.1], [0.1, 0]])
    d = make_dist_dict(mat)
    clusters = find_identity_clusters(d, nb_templates=2, threshold=0.2, min_cluster_size=3)
    assert clusters == []


# ---- Dummy template for TemplateIdentityFilter tests ----


class DummyTemplate:
    def __init__(self, arr):
        self.arr = arr

    def __eq__(self, other):
        return np.array_equal(self.arr, other.arr)


def make_templates(arrs):
    return [DummyTemplate(np.array(a)) for a in arrs]


# ---- Tests for TemplateIdentityFilter ----
class TestTemplateIdentityFilter:
    @pytest.fixture
    def sample_iris_templates(self, request):
        """Create sample IrisTemplate instances for testing.

        Can be parametrized with 'nb_templates' marker to specify number of templates.
        Default is 3 templates.
        """
        # Get number of templates from marker, default to 3
        nb_templates = getattr(request.node.get_closest_marker("nb_templates"), "args", [3])[0]

        iris_codes = [
            np.random.choice(2, size=(16, 200, 2)).astype(bool),  # Realistic size
            np.random.choice(2, size=(16, 200, 2)).astype(bool),
        ]
        mask_codes = [
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),  # Mostly valid
            np.random.choice(2, size=(16, 200, 2), p=[0.1, 0.9]).astype(bool),
        ]
        iris_code_version = "v2.1"

        return [
            IrisTemplate(
                iris_codes=iris_codes,
                mask_codes=mask_codes,
                iris_code_version=iris_code_version,
            )
            for _ in range(nb_templates)
        ]

    @pytest.fixture
    def sample_iris_templates_with_id(self, sample_iris_templates):
        return [
            IrisTemplateWithId.from_template(template, f"image_id_{i}")
            for i, template in enumerate(sample_iris_templates)
        ]

    def test_filter_removes_outlier(self, sample_iris_templates_with_id):
        mat = np.array([[0, 0.0, 1.0], [0.0, 0, 1.0], [1.0, 1.0, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.REMOVE
        )
        filtered = node.run(
            AlignedTemplates(
                templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
            )
        )
        assert filtered == sample_iris_templates_with_id[:2]

    def test_filter_raises_error(self, sample_iris_templates_with_id):
        mat = np.array([[0, 1.0], [1.0, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.RAISE_ERROR
        )
        with pytest.raises(Exception):
            at = AlignedTemplates(
                templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
            )
            node.run(at)

    def test_filter_logs_warning(self, monkeypatch, sample_iris_templates_with_id):
        mat = np.array([[0, 0.1, 0.5], [0.1, 0, 0.5], [0.5, 0.5, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.LOG_WARNING
        )
        called = {}

        def fake_warning(msg):
            called["msg"] = msg

        monkeypatch.setattr(logging, "warning", fake_warning)
        at = AlignedTemplates(
            templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
        )
        filtered = node.run(at)
        assert filtered == sample_iris_templates_with_id
        assert "exceed threshold" in called["msg"]

    @pytest.mark.nb_templates(4)
    def test_filter_min_templates_after_validation(self, sample_iris_templates_with_id):
        # 4 templates: 0,1 are a cluster; 2,3 are outliers
        mat = np.array([[0, 0.05, 0.8, 0.8], [0.05, 0, 0.8, 0.8], [0.8, 0.8, 0, 0.8], [0.8, 0.8, 0.8, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2,
            identity_validation_action=IdentityValidationAction.REMOVE,
            min_templates_after_validation=3,
        )
        with pytest.raises(E.IdentityValidationError):
            at = AlignedTemplates(
                templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
            )
            node.run(at)

    @pytest.mark.nb_templates(1)
    def test_filter_single_template(self, sample_iris_templates_with_id):
        node = TemplateIdentityFilter(identity_distance_threshold=0.2)
        at = AlignedTemplates(
            templates=sample_iris_templates_with_id, distances=DistanceMatrix(data={}), reference_template_id=0
        )
        filtered = node.run(at)
        assert filtered == sample_iris_templates_with_id

    @pytest.mark.nb_templates(4)
    def test_raises_error_if_two_clusters(self, sample_iris_templates_with_id):
        # 0,1 are close; 2,3 are close; between clusters is large
        mat = np.array([[0, 0.05, 0.81, 0.82], [0.05, 0, 0.79, 0.76], [0.81, 0.79, 0, 0.03], [0.82, 0.76, 0.03, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.REMOVE
        )
        with pytest.raises(E.IdentityValidationError):
            at = AlignedTemplates(
                templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
            )
            node.run(at)

    @pytest.mark.nb_templates(3)
    def test_raises_error_if_no_clusters(self, sample_iris_templates_with_id):
        mat = np.array([[0, 0.5, 0.6], [0.5, 0, 0.7], [0.6, 0.7, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.REMOVE
        )
        with pytest.raises(E.IdentityValidationError):
            at = AlignedTemplates(
                templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
            )
            node.run(at)

    @pytest.mark.nb_templates(3)
    def test_filter_all_outliers_raises(self, sample_iris_templates_with_id):
        # All templates are outliers (no clusters)
        mat = np.array([[0, 0.8, 0.8], [0.8, 0, 0.8], [0.8, 0.8, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2,
            identity_validation_action=IdentityValidationAction.REMOVE,
            min_templates_after_validation=1,
        )
        with pytest.raises(E.IdentityValidationError):
            at = AlignedTemplates(
                templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
            )
            node.run(at)

    @pytest.mark.nb_templates(2)
    def test_filter_log_warning_no_outliers(self, monkeypatch, sample_iris_templates_with_id):
        # All templates are within threshold, so no warning should be logged
        mat = np.array([[0, 0.1], [0.1, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.LOG_WARNING
        )
        called = {}

        def fake_warning(msg):
            called["msg"] = msg

        monkeypatch.setattr("logging.warning", fake_warning)
        at = AlignedTemplates(
            templates=sample_iris_templates_with_id, distances=DistanceMatrix(data=d), reference_template_id=0
        )
        filtered = node.run(at)
        assert filtered == sample_iris_templates_with_id
        assert "msg" not in called  # No warning should be logged
