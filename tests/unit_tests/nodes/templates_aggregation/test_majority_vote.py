import numpy as np
import pytest

from iris.io.dataclasses import IrisTemplate
from iris.nodes.templates_aggregation.majority_vote import MajorityVoteAggregation


@pytest.fixture
def mock_iris_template():
    """Create a mock IrisTemplate for testing."""
    iris_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
    mask_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
    return IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")


@pytest.fixture
def simple_templates():
    """Create simple test templates with known patterns."""
    # Create templates with predictable patterns for testing
    iris_codes_1 = [np.array([[[True, False], [False, True]], [[True, True], [False, False]]]).astype(bool)]
    mask_codes_1 = [np.array([[[True, True], [True, True]], [[True, True], [True, True]]]).astype(bool)]

    iris_codes_2 = [np.array([[[True, False], [True, True]], [[False, True], [False, False]]]).astype(bool)]
    mask_codes_2 = [np.array([[[True, True], [True, True]], [[True, True], [True, True]]]).astype(bool)]

    iris_codes_3 = [np.array([[[False, False], [True, True]], [[False, True], [True, False]]]).astype(bool)]
    mask_codes_3 = [np.array([[[True, True], [True, True]], [[True, True], [True, True]]]).astype(bool)]

    template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")
    template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")
    template3 = IrisTemplate(iris_codes=iris_codes_3, mask_codes=mask_codes_3, iris_code_version="v2.1")

    return [template1, template2, template3]


@pytest.fixture
def single_template():
    """Create a single template for testing edge cases."""
    iris_codes = [np.array([[[True, False], [False, True]]]).astype(bool)]
    mask_codes = [np.array([[[True, True], [True, True]]]).astype(bool)]
    return IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")


class TestMajorityVoteAggregation:
    """Test suite for MajorityVoteAggregation class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        aggregator = MajorityVoteAggregation()

        assert aggregator.params.consistency_threshold == 0.75
        assert aggregator.params.mask_threshold == 0.01
        assert aggregator.params.use_inconsistent_bits is True
        assert aggregator.params.inconsistent_bit_threshold == 0.4

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        aggregator = MajorityVoteAggregation(
            consistency_threshold=0.8, mask_threshold=0.6, use_inconsistent_bits=False, inconsistent_bit_threshold=0.3
        )

        assert aggregator.params.consistency_threshold == 0.8
        assert aggregator.params.mask_threshold == 0.6
        assert aggregator.params.use_inconsistent_bits is False
        assert aggregator.params.inconsistent_bit_threshold == 0.3

    def test_parameters_validation_consistency_threshold(self):
        """Test parameter validation for consistency_threshold."""
        # Valid values
        MajorityVoteAggregation(consistency_threshold=0.0)
        MajorityVoteAggregation(consistency_threshold=1.0)
        MajorityVoteAggregation(consistency_threshold=0.5)

        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            MajorityVoteAggregation(consistency_threshold=-0.1)

        with pytest.raises(Exception):  # Pydantic validation error
            MajorityVoteAggregation(consistency_threshold=1.1)

    def test_parameters_validation_mask_threshold(self):
        """Test parameter validation for mask_threshold."""
        # Valid values
        MajorityVoteAggregation(mask_threshold=0.0)
        MajorityVoteAggregation(mask_threshold=1.0)
        MajorityVoteAggregation(mask_threshold=0.5)

        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            MajorityVoteAggregation(mask_threshold=-0.1)

        with pytest.raises(Exception):  # Pydantic validation error
            MajorityVoteAggregation(mask_threshold=1.1)

    def test_parameters_validation_inconsistent_bit_threshold(self):
        """Test parameter validation for inconsistent_bit_threshold."""
        # Valid values
        MajorityVoteAggregation(inconsistent_bit_threshold=0.0)
        MajorityVoteAggregation(inconsistent_bit_threshold=1.0)
        MajorityVoteAggregation(inconsistent_bit_threshold=0.5)

        # Invalid values should raise validation error
        with pytest.raises(Exception):  # Pydantic validation error
            MajorityVoteAggregation(inconsistent_bit_threshold=-0.1)

        with pytest.raises(Exception):  # Pydantic validation error
            MajorityVoteAggregation(inconsistent_bit_threshold=1.1)

    def test_run_empty_templates_list(self):
        """Test run method with empty templates list."""
        aggregator = MajorityVoteAggregation()

        with pytest.raises(ValueError, match="No templates provided for combination"):
            aggregator.run([])

    def test_run_single_template(self, single_template):
        """Test run method with single template."""
        aggregator = MajorityVoteAggregation()

        weighted_template = aggregator.run([single_template])
        combined_template = weighted_template.as_iris_template()
        weights = weighted_template.weights

        # Should return the same template
        assert weighted_template.iris_code_version == single_template.iris_code_version
        assert len(combined_template.iris_codes) == len(single_template.iris_codes)
        assert len(combined_template.mask_codes) == len(single_template.mask_codes)

        # Weights should be all ones
        assert len(weights) == len(single_template.iris_codes)
        np.testing.assert_array_equal(weights[0], np.ones_like(single_template.iris_codes[0]))

    def test_run_multiple_templates(self, simple_templates):
        """Test run method with multiple templates."""
        aggregator = MajorityVoteAggregation(
            consistency_threshold=0.5, mask_threshold=0.5, use_inconsistent_bits=True, inconsistent_bit_threshold=0.3
        )

        weighted_template = aggregator.run(simple_templates)
        combined_template = weighted_template.as_iris_template()
        weights = weighted_template.weights

        # Check basic structure
        assert isinstance(combined_template, IrisTemplate)
        assert combined_template.iris_code_version == simple_templates[0].iris_code_version
        assert len(combined_template.iris_codes) == len(simple_templates[0].iris_codes)
        assert len(combined_template.mask_codes) == len(simple_templates[0].mask_codes)
        assert len(weights) == len(simple_templates[0].iris_codes)

        # Check shapes
        for i in range(len(combined_template.iris_codes)):
            assert combined_template.iris_codes[i].shape == simple_templates[0].iris_codes[i].shape
            assert combined_template.mask_codes[i].shape == simple_templates[0].mask_codes[i].shape
            assert weights[i].shape == simple_templates[0].iris_codes[i].shape

    def test_combine_templates_method(self, simple_templates):
        """Test combine_templates method directly."""
        aggregator = MajorityVoteAggregation()

        weighted_template = aggregator.combine_templates(simple_templates)
        combined_template = weighted_template.as_iris_template()
        weights = weighted_template.weights

        assert isinstance(combined_template, IrisTemplate)
        assert isinstance(weights, list)
        assert len(weights) == len(simple_templates[0].iris_codes)

    def test_combine_wavelet_codes_majority_voting(self):
        """Test _combine_wavelet_codes method with known patterns."""
        aggregator = MajorityVoteAggregation(
            consistency_threshold=0.5, mask_threshold=0.5, use_inconsistent_bits=True, inconsistent_bit_threshold=0.3
        )

        # Create test data where majority voting is clear
        iris_codes = [
            np.array([[[True, False]], [[False, True]]]).astype(bool),  # Template 1
            np.array([[[True, False]], [[False, True]]]).astype(bool),  # Template 2
            np.array([[[False, True]], [[True, False]]]).astype(bool),  # Template 3
        ]

        mask_codes = [
            np.array([[[True, True]], [[True, True]]]).astype(bool),  # Template 1
            np.array([[[True, True]], [[True, True]]]).astype(bool),  # Template 2
            np.array([[[True, True]], [[True, True]]]).astype(bool),  # Template 3
        ]

        combined_iris, combined_mask, weight = aggregator._combine_wavelet_codes(iris_codes, mask_codes)

        # Check majority voting results
        # Position [0,0,0]: True appears 2/3 times -> should be True
        # Position [0,0,1]: False appears 2/3 times -> should be False
        # Position [1,0,0]: False appears 2/3 times -> should be False
        # Position [1,0,1]: True appears 2/3 times -> should be True
        assert bool(combined_iris[0, 0, 0]) is True
        assert bool(combined_iris[0, 0, 1]) is False
        assert bool(combined_iris[1, 0, 0]) is False
        assert bool(combined_iris[1, 0, 1]) is True

        # All masks should be True since all templates have valid masks
        np.testing.assert_array_equal(combined_mask, np.ones_like(combined_mask))

    def test_combine_wavelet_codes_mask_threshold(self):
        """Test _combine_wavelet_codes method with mask threshold."""
        aggregator = MajorityVoteAggregation(mask_threshold=0.8)  # High threshold

        iris_codes = [
            np.array([[[True, False]]]).astype(bool),
            np.array([[[True, False]]]).astype(bool),
            np.array([[[True, False]]]).astype(bool),
        ]

        # Only 2 out of 3 templates have valid masks at position [0,0,1]
        mask_codes = [
            np.array([[[True, True]]]).astype(bool),
            np.array([[[True, False]]]).astype(bool),  # Invalid at [0,0,1]
            np.array([[[True, True]]]).astype(bool),
        ]

        _, combined_mask, __ = aggregator._combine_wavelet_codes(iris_codes, mask_codes)

        # Position [0,0,0] should be valid (3/3 = 1.0 >= 0.8)
        # Position [0,0,1] should be invalid (2/3 = 0.67 < 0.8)
        assert bool(combined_mask[0, 0, 0]) is True
        assert bool(combined_mask[0, 0, 1]) is False

    def test_combine_wavelet_codes_consistency_weights(self):
        """Test _combine_wavelet_codes method weight calculation."""
        aggregator = MajorityVoteAggregation(
            consistency_threshold=0.6, use_inconsistent_bits=True, inconsistent_bit_threshold=0.3
        )

        # Create data with different consistency levels
        iris_codes = [
            np.array([[[True, False]], [[True, True]]]).astype(bool),  # High consistency at [0,0,0]
            np.array([[[True, True]], [[True, False]]]).astype(bool),  # Low consistency at [0,0,1]
            np.array([[[True, False]], [[False, False]]]).astype(bool),  # Medium consistency
        ]

        mask_codes = [
            np.array([[[True, True]], [[True, True]]]).astype(bool),
            np.array([[[True, True]], [[True, True]]]).astype(bool),
            np.array([[[True, True]], [[True, True]]]).astype(bool),
        ]

        combined_iris, combined_mask, weight = aggregator._combine_wavelet_codes(iris_codes, mask_codes)

        # Position [0,0,0]: 3/3 = 1.0 vote fraction -> consistency = |1.0 - 0.5| * 2 = 1.0
        # Should get high weight (>= consistency_threshold)
        assert weight[0, 0, 0] >= aggregator.params.consistency_threshold

        # Position [0,0,1]: 1/3 = 0.33 vote fraction -> consistency = |0.33 - 0.5| * 2 = 0.33
        # Should get inconsistent bit weight (< consistency_threshold)
        assert weight[0, 0, 1] == aggregator.params.inconsistent_bit_threshold

    def test_combine_wavelet_codes_no_inconsistent_bits(self):
        """Test _combine_wavelet_codes method with inconsistent bits disabled."""
        aggregator = MajorityVoteAggregation(consistency_threshold=0.8, use_inconsistent_bits=False)

        iris_codes = [
            np.array([[[True, False]]]).astype(bool),
            np.array([[[False, True]]]).astype(bool),
            np.array([[[True, False]]]).astype(bool),
        ]

        mask_codes = [
            np.array([[[True, True]]]).astype(bool),
            np.array([[[True, True]]]).astype(bool),
            np.array([[[True, True]]]).astype(bool),
        ]

        combined_iris, combined_mask, weight = aggregator._combine_wavelet_codes(iris_codes, mask_codes)

        # Low consistency positions should get weight 0 when inconsistent bits disabled
        # Position [0,0,1]: 1/3 = 0.33 vote fraction -> consistency = 0.33 < 0.8
        assert weight[0, 0, 1] == 0.0

    def test_run_with_different_shapes(self):
        """Test run method with templates having different shapes."""
        iris_codes_1 = [np.random.choice(2, size=(4, 8, 2)).astype(bool)]
        mask_codes_1 = [np.random.choice(2, size=(4, 8, 2)).astype(bool)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        iris_codes_2 = [np.random.choice(2, size=(6, 10, 2)).astype(bool)]  # Different shape
        mask_codes_2 = [np.random.choice(2, size=(6, 10, 2)).astype(bool)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        aggregator = MajorityVoteAggregation()

        # This should raise an error due to mismatched shapes
        with pytest.raises(Exception):  # Could be ValueError or similar
            aggregator.run([template1, template2])

    def test_parameters_type_attribute(self):
        """Test that __parameters_type__ is correctly set."""
        assert MajorityVoteAggregation.__parameters_type__ == MajorityVoteAggregation.Parameters

    def test_algorithm_inheritance(self):
        """Test that MajorityVoteAggregation properly inherits from Algorithm."""
        from iris.io.class_configs import Algorithm

        aggregator = MajorityVoteAggregation()
        assert isinstance(aggregator, Algorithm)
        assert hasattr(aggregator, "params")
        assert hasattr(aggregator, "run")
        assert hasattr(aggregator, "execute")

    def test_edge_case_all_zeros_iris_codes(self):
        """Test with templates where all iris codes are False."""
        iris_codes_1 = [np.zeros((2, 2, 2), dtype=bool)]
        mask_codes_1 = [np.ones((2, 2, 2), dtype=bool)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        iris_codes_2 = [np.zeros((2, 2, 2), dtype=bool)]
        mask_codes_2 = [np.ones((2, 2, 2), dtype=bool)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        aggregator = MajorityVoteAggregation()
        weighted_template = aggregator.run([template1, template2])
        combined_template = weighted_template.as_iris_template()

        # All iris codes should remain False
        np.testing.assert_array_equal(combined_template.iris_codes[0], np.zeros((2, 2, 2), dtype=bool))
        # All masks should be True
        np.testing.assert_array_equal(combined_template.mask_codes[0], np.ones((2, 2, 2), dtype=bool))

    def test_edge_case_all_ones_iris_codes(self):
        """Test with templates where all iris codes are True."""
        iris_codes_1 = [np.ones((2, 2, 2), dtype=bool)]
        mask_codes_1 = [np.ones((2, 2, 2), dtype=bool)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        iris_codes_2 = [np.ones((2, 2, 2), dtype=bool)]
        mask_codes_2 = [np.ones((2, 2, 2), dtype=bool)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        aggregator = MajorityVoteAggregation()
        weighted_template = aggregator.run([template1, template2])
        combined_template = weighted_template.as_iris_template()

        # All iris codes should remain True
        np.testing.assert_array_equal(combined_template.iris_codes[0], np.ones((2, 2, 2), dtype=bool))
        # All masks should be True
        np.testing.assert_array_equal(combined_template.mask_codes[0], np.ones((2, 2, 2), dtype=bool))

    def test_edge_case_all_invalid_masks(self):
        """Test with templates where all masks are False."""
        iris_codes_1 = [np.random.choice(2, size=(2, 2, 2)).astype(bool)]
        mask_codes_1 = [np.zeros((2, 2, 2), dtype=bool)]  # All invalid
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        iris_codes_2 = [np.random.choice(2, size=(2, 2, 2)).astype(bool)]
        mask_codes_2 = [np.zeros((2, 2, 2), dtype=bool)]  # All invalid
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        aggregator = MajorityVoteAggregation(mask_threshold=0.5)
        weighted_template = aggregator.run([template1, template2])
        combined_template = weighted_template.as_iris_template()

        # All masks should be False due to mask_threshold
        np.testing.assert_array_equal(combined_template.mask_codes[0], np.zeros((2, 2, 2), dtype=bool))
