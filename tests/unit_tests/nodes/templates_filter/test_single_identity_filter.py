import logging

import numpy as np
import pytest

from iris.nodes.templates_filter.single_identity_filter import (
    IdentityValidationAction,
    TemplateIdentityFilter,
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
    assert greedy_purification(d, threshold=0.2, n=3) == []


def test_greedy_one_outlier():
    mat = np.array([[0, 0.1, 0.5], [0.1, 0, 0.5], [0.5, 0.5, 0]])
    d = make_dist_dict(mat)
    assert greedy_purification(d, threshold=0.2, n=3) == [2]


def test_greedy_multiple_outliers():
    mat = np.array([[0, 0.6, 0.6, 0.1], [0.6, 0, 0.6, 0.1], [0.6, 0.6, 0, 0.1], [0.1, 0.1, 0.1, 0]])
    d = make_dist_dict(mat)
    removed = greedy_purification(d, threshold=0.2, n=4)
    assert set(removed) == {0, 1}


def test_greedy_min_templates_respected():
    mat = np.array([[0, 0.9, 0.9], [0.9, 0, 0.9], [0.9, 0.9, 0]])
    d = make_dist_dict(mat)
    assert greedy_purification(d, threshold=0.2, n=3, min_templates=1) == [0, 1]
    assert greedy_purification(d, threshold=0.2, n=3, min_templates=2) == [0]


def test_greedy_single_template():
    d = {}
    assert greedy_purification(d, threshold=0.2, n=1) == []


def test_greedy_empty_input():
    assert greedy_purification({}, threshold=0.2, n=0) == []


def test_greedy_tie_breaking():
    mat = np.array([[0, 0.5, 0.5], [0.5, 0, 0.1], [0.5, 0.1, 0]])
    d = make_dist_dict(mat)
    removed = greedy_purification(d, threshold=0.2, n=3)
    assert removed == [0] or removed == [1]


def test_greedy_two_clusters():
    # 0,1 are close; 2,3 are close; between clusters is large
    mat = np.array([[0, 0.05, 0.8, 0.8], [0.05, 0, 0.8, 0.8], [0.8, 0.8, 0, 0.05], [0.8, 0.8, 0.05, 0]])
    d = make_dist_dict(mat)
    removed = greedy_purification(d, threshold=0.2, n=4)
    # Only one cluster should remain, so either [0,1] or [2,3] should be removed
    assert set(removed) == {0, 1} or set(removed) == {2, 3}


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
    def test_filter_removes_outlier(self):
        arrs = [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
        templates = make_templates(arrs)
        mat = np.array([[0, 0.0, 1.0], [0.0, 0, 1.0], [1.0, 1.0, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.REMOVE
        )
        filtered = node.run(templates, pairwise_distances=d)
        assert filtered == templates[:2]

    def test_filter_raises_error(self):
        arrs = [[0, 0, 0], [1, 1, 1]]
        templates = make_templates(arrs)
        mat = np.array([[0, 1.0], [1.0, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.RAISE_ERROR
        )
        with pytest.raises(Exception):
            node.run(templates, pairwise_distances=d)

    def test_filter_logs_warning(self, monkeypatch):
        arrs = [[0, 0, 0], [1, 1, 1]]
        templates = make_templates(arrs)
        mat = np.array([[0, 1.0], [1.0, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.LOG_WARNING
        )
        called = {}

        def fake_warning(msg):
            called["msg"] = msg

        monkeypatch.setattr(logging, "warning", fake_warning)
        filtered = node.run(templates, pairwise_distances=d)
        assert filtered == templates
        assert "exceed threshold" in called["msg"]

    def test_filter_min_templates_after_validation(self):
        arrs = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
        templates = make_templates(arrs)
        mat = np.array([[0, 1.0, 1.0], [1.0, 0, 1.0], [1.0, 1.0, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2,
            identity_validation_action=IdentityValidationAction.REMOVE,
            min_templates_after_validation=2,
        )
        with pytest.raises(Exception):
            node.run(templates, pairwise_distances=d)

    def test_filter_computes_distances_if_not_provided(self, monkeypatch):
        arrs = [[0, 0, 0], [0, 0, 0], [1, 1, 1]]
        templates = make_templates(arrs)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.REMOVE
        )
        # Patch where used, not where defined!
        monkeypatch.setattr(
            "iris.nodes.templates_filter.single_identity_filter.simple_hamming_distance",
            lambda a, b: (1.0 if not np.array_equal(a.arr, b.arr) else 0.0, None),
        )
        filtered = node.run(templates)
        assert len(filtered) == 2

    def test_filter_single_template(self):
        arrs = [[0, 0, 0]]
        templates = make_templates(arrs)
        node = TemplateIdentityFilter(identity_distance_threshold=0.2)
        filtered = node.run(templates)
        assert filtered == templates

    def test_filter_two_clusters(self):
        # 0,1 are close; 2,3 are close; between clusters is large
        arrs = [[0, 0, 0], [0, 0, 1], [1, 1, 1], [1, 1, 2]]
        templates = make_templates(arrs)
        mat = np.array([[0, 0.05, 0.81, 0.82], [0.05, 0, 0.79, 0.76], [0.81, 0.79, 0, 0.03], [0.82, 0.76, 0.03, 0]])
        d = make_dist_dict(mat)
        node = TemplateIdentityFilter(
            identity_distance_threshold=0.2, identity_validation_action=IdentityValidationAction.REMOVE
        )
        filtered = node.run(templates, pairwise_distances=d)
        # Only one cluster should remain, so either [0,1] or [2,3] should be left
        assert filtered == templates[:2] or filtered == templates[2:]
