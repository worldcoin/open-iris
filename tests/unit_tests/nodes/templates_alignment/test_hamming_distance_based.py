from unittest.mock import patch

import numpy as np
import pytest

from iris.io.dataclasses import IrisTemplate
from iris.nodes.templates_alignment.hamming_distance_based import (
    HammingDistanceBasedAlignment,
    ReferenceSelectionMethod,
)


class TestHammingDistanceBasedAlignment:
    """Test cases for HammingDistanceBasedAlignment class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        alignment = HammingDistanceBasedAlignment()

        assert alignment.params.rotation_shift == 15
        assert alignment.params.use_first_as_reference is False
        assert alignment.params.normalise is True
        assert alignment.params.reference_selection_method == ReferenceSelectionMethod.LINEAR

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        alignment = HammingDistanceBasedAlignment(
            rotation_shift=10,
            use_first_as_reference=True,
            normalise=False,
            reference_selection_method=ReferenceSelectionMethod.MEAN_SQUARED,
        )

        assert alignment.params.rotation_shift == 10
        assert alignment.params.use_first_as_reference is True
        assert alignment.params.normalise is False
        assert alignment.params.reference_selection_method == ReferenceSelectionMethod.MEAN_SQUARED

    def test_run_empty_templates_list(self):
        """Test that empty templates list raises ValueError."""
        alignment = HammingDistanceBasedAlignment()

        with pytest.raises(ValueError, match="No templates provided for alignment"):
            alignment.run([])

    def test_run_single_template(self):
        """Test that single template returns the same template."""
        alignment = HammingDistanceBasedAlignment()

        template = IrisTemplate(
            iris_codes=[np.array([[[True, False], [False, True]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )

        result = alignment.run([template])

        assert len(result) == 1
        assert result[0] == template

    def test_run_multiple_templates_first_as_reference(self):
        """Test alignment with multiple templates using first as reference."""
        alignment = HammingDistanceBasedAlignment(use_first_as_reference=True)

        # Create mock templates
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False], [False, True]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True], [True, False]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )

        with patch.object(alignment, "_find_optimal_rotation", return_value=1) as mock_rotation, patch.object(
            alignment, "_rotate_template", return_value=template2
        ) as mock_rotate:
            result = alignment.run([template1, template2])

            assert len(result) == 2
            assert result[0] == template1  # Reference template unchanged
            mock_rotation.assert_called_once_with(template2, template1)
            mock_rotate.assert_called_once_with(template2, 1)

    def test_run_multiple_templates_best_reference(self):
        """Test alignment with multiple templates using best reference."""
        alignment = HammingDistanceBasedAlignment(use_first_as_reference=False)

        # Create mock templates
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False], [False, True]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True], [True, False]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )

        with patch.object(alignment, "_find_best_reference", return_value=1) as mock_best_ref, patch.object(
            alignment, "_find_optimal_rotation", return_value=2
        ) as mock_rotation, patch.object(alignment, "_rotate_template", return_value=template1) as mock_rotate:
            result = alignment.run([template1, template2])

            assert len(result) == 2
            mock_best_ref.assert_called_once_with([template1, template2])
            mock_rotation.assert_called_once_with(template1, template2)
            mock_rotate.assert_called_once_with(template1, 2)

    def test_find_best_reference(self):
        """Test finding the best reference template."""
        alignment = HammingDistanceBasedAlignment()

        # Create mock templates
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False], [False, True]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True], [True, False]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )
        template3 = IrisTemplate(
            iris_codes=[np.array([[[True, True], [False, False]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )

        # Mock the hamming distance calculations
        # Template 1: distances [0.5, 0.7], sum = 1.2
        # Template 2: distances [0.5, 0.3], sum = 0.8 (best)
        # Template 3: distances [0.7, 0.3], sum = 1.0
        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.5, 0),  # template1 vs template2
                (0.7, 0),  # template1 vs template3
                (0.5, 0),  # template2 vs template1
                (0.3, 0),  # template2 vs template3
                (0.7, 0),  # template3 vs template1
                (0.3, 0),  # template3 vs template2
            ]

            best_idx = alignment._find_best_reference([template1, template2, template3])

            assert best_idx == 1  # Template 2 has the minimum sum of distances

    def test_find_optimal_rotation(self):
        """Test finding optimal rotation for template alignment."""
        alignment = HammingDistanceBasedAlignment()

        template = IrisTemplate(
            iris_codes=[np.array([[[True, False], [False, True]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )
        reference = IrisTemplate(
            iris_codes=[np.array([[[False, True], [True, False]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.return_value = (0.3, 5)

            rotation = alignment._find_optimal_rotation(template, reference)

            assert rotation == 5
            mock_hd.assert_called_once_with(template, reference, rotation_shift=15, normalise=True)

    def test_normalise_parameter_usage(self):
        """Test that normalise parameter is properly passed to hamming distance calculations."""
        alignment = HammingDistanceBasedAlignment(normalise=True)

        template = IrisTemplate(
            iris_codes=[np.array([[[True, False], [False, True]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )
        reference = IrisTemplate(
            iris_codes=[np.array([[[False, True], [True, False]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.return_value = (0.3, 5)

            # Test _find_optimal_rotation uses normalise parameter
            rotation = alignment._find_optimal_rotation(template, reference)

            assert rotation == 5
            mock_hd.assert_called_with(template, reference, rotation_shift=15, normalise=True)

            # Test _find_best_reference uses normalise parameter
            mock_hd.reset_mock()
            mock_hd.side_effect = [(0.3, 0), (0.2, 0)]  # Two calls for _find_best_reference

            _ = alignment._find_best_reference([template, reference])

            # Verify that normalise=True was used in both calls
            for call in mock_hd.call_args_list:
                assert call[1]["normalise"] is True

    def test_rotate_template_zero_rotation(self):
        """Test that zero rotation returns the original template."""
        alignment = HammingDistanceBasedAlignment()

        template = IrisTemplate(
            iris_codes=[np.array([[[True, False], [False, True]]])],
            mask_codes=[np.array([[[True, True], [True, True]]])],
            iris_code_version="v2.1",
        )

        result = alignment._rotate_template(template, 0)

        assert result == template

    def test_rotate_template_positive_rotation(self):
        """Test template rotation with positive offset."""
        alignment = HammingDistanceBasedAlignment()

        # Create a template with a specific pattern
        iris_code = np.array([[[True, False], [False, True], [True, True]]])
        mask_code = np.array([[[True, True], [False, True], [True, False]]])

        template = IrisTemplate(iris_codes=[iris_code], mask_codes=[mask_code], iris_code_version="v2.1")

        result = alignment._rotate_template(template, 1)

        # Check that the arrays were rolled by 1 position
        expected_iris = np.roll(iris_code, 1, axis=1)
        expected_mask = np.roll(mask_code, 1, axis=1)

        np.testing.assert_array_equal(result.iris_codes[0], expected_iris)
        np.testing.assert_array_equal(result.mask_codes[0], expected_mask)
        assert result.iris_code_version == template.iris_code_version

    def test_rotate_template_negative_rotation(self):
        """Test template rotation with negative offset."""
        alignment = HammingDistanceBasedAlignment()

        # Create a template with a specific pattern
        iris_code = np.array([[[True, False], [False, True], [True, True]]])
        mask_code = np.array([[[True, True], [False, True], [True, False]]])

        template = IrisTemplate(iris_codes=[iris_code], mask_codes=[mask_code], iris_code_version="v2.1")

        result = alignment._rotate_template(template, -1)

        # Check that the arrays were rolled by -1 position
        expected_iris = np.roll(iris_code, -1, axis=1)
        expected_mask = np.roll(mask_code, -1, axis=1)

        np.testing.assert_array_equal(result.iris_codes[0], expected_iris)
        np.testing.assert_array_equal(result.mask_codes[0], expected_mask)
        assert result.iris_code_version == template.iris_code_version

    def test_rotate_template_multiple_wavelets(self):
        """Test template rotation with multiple wavelets."""
        alignment = HammingDistanceBasedAlignment()

        # Create a template with multiple wavelets
        iris_code1 = np.array([[[True, False], [False, True]]])
        iris_code2 = np.array([[[False, True], [True, False]]])
        mask_code1 = np.array([[[True, True], [False, True]]])
        mask_code2 = np.array([[[False, False], [True, True]]])

        template = IrisTemplate(
            iris_codes=[iris_code1, iris_code2], mask_codes=[mask_code1, mask_code2], iris_code_version="v2.1"
        )

        result = alignment._rotate_template(template, 1)

        # Check that all wavelets were rotated
        expected_iris1 = np.roll(iris_code1, 1, axis=1)
        expected_iris2 = np.roll(iris_code2, 1, axis=1)
        expected_mask1 = np.roll(mask_code1, 1, axis=1)
        expected_mask2 = np.roll(mask_code2, 1, axis=1)

        np.testing.assert_array_equal(result.iris_codes[0], expected_iris1)
        np.testing.assert_array_equal(result.iris_codes[1], expected_iris2)
        np.testing.assert_array_equal(result.mask_codes[0], expected_mask1)
        np.testing.assert_array_equal(result.mask_codes[1], expected_mask2)
        assert result.iris_code_version == template.iris_code_version

    def test_aggregate_distances_linear(self):
        """Test linear distance aggregation (sum)."""
        alignment = HammingDistanceBasedAlignment(reference_selection_method=ReferenceSelectionMethod.LINEAR)

        distances = [0.2, 0.3, 0.5]
        result = alignment._aggregate_distances(distances)

        assert result == 1.0  # sum of distances

    def test_aggregate_distances_mean_squared(self):
        """Test mean squared distance aggregation."""
        alignment = HammingDistanceBasedAlignment(reference_selection_method=ReferenceSelectionMethod.MEAN_SQUARED)

        distances = [0.2, 0.4, 0.6]
        result = alignment._aggregate_distances(distances)

        expected = np.mean(np.array(distances) ** 2)
        assert result == pytest.approx(expected)

    def test_aggregate_distances_root_mean_squared(self):
        """Test root mean squared distance aggregation."""
        alignment = HammingDistanceBasedAlignment(reference_selection_method=ReferenceSelectionMethod.ROOT_MEAN_SQUARED)

        distances = [0.3, 0.4, 0.5]
        result = alignment._aggregate_distances(distances)

        expected = np.sqrt(np.mean(np.array(distances) ** 2))
        assert result == pytest.approx(expected)

    def test_aggregate_distances_empty_list(self):
        """Test distance aggregation with empty list."""
        alignment = HammingDistanceBasedAlignment()

        result = alignment._aggregate_distances([])

        assert result == 0.0

    def test_aggregate_distances_invalid_method(self):
        """Test distance aggregation with invalid method."""
        alignment = HammingDistanceBasedAlignment()

        # Create a mock parameters object with invalid method
        from unittest.mock import Mock

        mock_params = Mock()
        mock_params.reference_selection_method = "invalid_method"

        # Temporarily replace params
        original_params = alignment.params
        alignment.params = mock_params

        try:
            with pytest.raises(ValueError, match="Unknown reference selection method"):
                alignment._aggregate_distances([0.1, 0.2])
        finally:
            # Restore original params
            alignment.params = original_params

    def test_find_best_reference_mean_squared(self):
        """Test finding best reference using mean squared method."""
        alignment = HammingDistanceBasedAlignment(reference_selection_method=ReferenceSelectionMethod.MEAN_SQUARED)

        # Create mock templates
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template3 = IrisTemplate(
            iris_codes=[np.array([[[True, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )

        # Mock hamming distance calculations
        # Template 1: distances [0.2, 0.4], mean_squared = (0.04 + 0.16) / 2 = 0.1
        # Template 2: distances [0.2, 0.1], mean_squared = (0.04 + 0.01) / 2 = 0.025 (best)
        # Template 3: distances [0.4, 0.1], mean_squared = (0.16 + 0.01) / 2 = 0.085
        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.2, 0),  # template1 vs template2
                (0.4, 0),  # template1 vs template3
                (0.2, 0),  # template2 vs template1
                (0.1, 0),  # template2 vs template3
                (0.4, 0),  # template3 vs template1
                (0.1, 0),  # template3 vs template2
            ]

            best_idx = alignment._find_best_reference([template1, template2, template3])

            assert best_idx == 1  # Template 2 has the minimum mean squared distance

    def test_find_best_reference_root_mean_squared(self):
        """Test finding best reference using root mean squared method."""
        alignment = HammingDistanceBasedAlignment(reference_selection_method=ReferenceSelectionMethod.ROOT_MEAN_SQUARED)

        # Create mock templates
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )

        # Mock hamming distance calculations
        # Template 1: distances [0.3], rms = sqrt(0.09) = 0.3
        # Template 2: distances [0.3], rms = sqrt(0.09) = 0.3 (same, first will be selected)
        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.3, 0),  # template1 vs template2
                (0.3, 0),  # template2 vs template1
            ]

            best_idx = alignment._find_best_reference([template1, template2])

            assert best_idx == 0  # First template selected when distances are equal

    def test_reference_selection_methods_numerical_comparison(self):
        """Test reference selection with numerical example showing method differences."""
        # Create simple templates for clear numerical demonstration
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template3 = IrisTemplate(
            iris_codes=[np.array([[[True, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )

        # Define distance matrix for clear numerical example:
        # Template 1 distances to others: [0.2, 0.6]
        # Template 2 distances to others: [0.2, 0.3]
        # Template 3 distances to others: [0.6, 0.3]

        templates = [template1, template2, template3]

        # Test LINEAR method (sum of distances)
        alignment_linear = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.LINEAR, use_first_as_reference=False
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.2, 0),
                (0.6, 0),  # template1 distances
                (0.2, 0),
                (0.3, 0),  # template2 distances
                (0.6, 0),
                (0.3, 0),  # template3 distances
            ]

            best_idx_linear = alignment_linear._find_best_reference(templates)

            # Expected calculations:
            # Template 1: sum = 0.2 + 0.6 = 0.8
            # Template 2: sum = 0.2 + 0.3 = 0.5 (winner)
            # Template 3: sum = 0.6 + 0.3 = 0.9
            assert best_idx_linear == 1

        # Test MEAN_SQUARED method
        alignment_mean_squared = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.MEAN_SQUARED, use_first_as_reference=False
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.2, 0),
                (0.6, 0),  # template1 distances
                (0.2, 0),
                (0.3, 0),  # template2 distances
                (0.6, 0),
                (0.3, 0),  # template3 distances
            ]

            best_idx_mean_squared = alignment_mean_squared._find_best_reference(templates)

            # Expected calculations:
            # Template 1: mean_squared = (0.2² + 0.6²) / 2 = (0.04 + 0.36) / 2 = 0.20
            # Template 2: mean_squared = (0.2² + 0.3²) / 2 = (0.04 + 0.09) / 2 = 0.065 (winner)
            # Template 3: mean_squared = (0.6² + 0.3²) / 2 = (0.36 + 0.09) / 2 = 0.225
            assert best_idx_mean_squared == 1

        # Test ROOT_MEAN_SQUARED method
        alignment_rms = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.ROOT_MEAN_SQUARED, use_first_as_reference=False
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.2, 0),
                (0.6, 0),  # template1 distances
                (0.2, 0),
                (0.3, 0),  # template2 distances
                (0.6, 0),
                (0.3, 0),  # template3 distances
            ]

            best_idx_rms = alignment_rms._find_best_reference(templates)

            # Expected calculations:
            # Template 1: rms = sqrt((0.2² + 0.6²) / 2) = sqrt(0.20) ≈ 0.447
            # Template 2: rms = sqrt((0.2² + 0.3²) / 2) = sqrt(0.065) ≈ 0.255 (winner)
            # Template 3: rms = sqrt((0.6² + 0.3²) / 2) = sqrt(0.225) ≈ 0.474
            assert best_idx_rms == 1

        # Verify the actual aggregation calculations
        distances_template1 = [0.2, 0.6]
        distances_template2 = [0.2, 0.3]

        # Test aggregation methods directly
        assert alignment_linear._aggregate_distances(distances_template1) == 0.8
        assert alignment_linear._aggregate_distances(distances_template2) == 0.5

        assert alignment_mean_squared._aggregate_distances(distances_template1) == pytest.approx(0.20)
        assert alignment_mean_squared._aggregate_distances(distances_template2) == pytest.approx(0.065)

        assert alignment_rms._aggregate_distances(distances_template1) == pytest.approx(0.447, abs=0.001)
        assert alignment_rms._aggregate_distances(distances_template2) == pytest.approx(0.255, abs=0.001)

    def test_reference_selection_methods_different_winners(self):
        """Test case where different reference selection methods choose different templates."""
        # Create 4 templates to increase chance of different winners
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template3 = IrisTemplate(
            iris_codes=[np.array([[[True, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template4 = IrisTemplate(
            iris_codes=[np.array([[[False, False]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )

        # Carefully designed distance matrix where different methods favor different templates:
        # Template 1 distances to others: [0.8, 0.9, 0.1] -> sum=1.8, mean_squared=0.55, rms=0.74
        # Template 2 distances to others: [0.8, 0.2, 0.9] -> sum=1.9, mean_squared=0.63, rms=0.79
        # Template 3 distances to others: [0.9, 0.2, 0.5] -> sum=1.6, mean_squared=0.37, rms=0.61 (LINEAR winner)
        # Template 4 distances to others: [0.1, 0.9, 0.5] -> sum=1.5, mean_squared=0.357, rms=0.60 (MEAN_SQUARED/RMS winner)

        templates = [template1, template2, template3, template4]

        # Test LINEAR method - should prefer template 4 (sum=1.5)
        alignment_linear = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.LINEAR, use_first_as_reference=False
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.8, 0),
                (0.9, 0),
                (0.1, 0),  # template1 distances
                (0.8, 0),
                (0.2, 0),
                (0.9, 0),  # template2 distances
                (0.9, 0),
                (0.2, 0),
                (0.5, 0),  # template3 distances
                (0.1, 0),
                (0.9, 0),
                (0.5, 0),  # template4 distances
            ]

            best_idx_linear = alignment_linear._find_best_reference(templates)
            assert best_idx_linear == 3  # Template 4 wins with sum=1.5

        # Test MEAN_SQUARED method - should prefer template 4 (mean_squared≈0.357)
        alignment_mean_squared = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.MEAN_SQUARED, use_first_as_reference=False
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.8, 0),
                (0.9, 0),
                (0.1, 0),  # template1 distances
                (0.8, 0),
                (0.2, 0),
                (0.9, 0),  # template2 distances
                (0.9, 0),
                (0.2, 0),
                (0.5, 0),  # template3 distances
                (0.1, 0),
                (0.9, 0),
                (0.5, 0),  # template4 distances
            ]

            best_idx_mean_squared = alignment_mean_squared._find_best_reference(templates)
            # Template 4: mean_squared = (0.1² + 0.9² + 0.5²) / 3 = (0.01 + 0.81 + 0.25) / 3 = 0.357
            # Template 3: mean_squared = (0.9² + 0.2² + 0.5²) / 3 = (0.81 + 0.04 + 0.25) / 3 = 0.367
            assert best_idx_mean_squared == 3  # Template 4 wins

        # Verify the direct calculations to show the differences clearly
        distances_t1 = [0.8, 0.9, 0.1]
        distances_t3 = [0.9, 0.2, 0.5]
        distances_t4 = [0.1, 0.9, 0.5]

        # LINEAR sums
        assert alignment_linear._aggregate_distances(distances_t1) == pytest.approx(1.8)
        assert alignment_linear._aggregate_distances(distances_t3) == pytest.approx(1.6)
        assert alignment_linear._aggregate_distances(distances_t4) == pytest.approx(1.5)  # Winner

        # MEAN_SQUARED values
        expected_ms_t1 = (0.8**2 + 0.9**2 + 0.1**2) / 3  # ≈ 0.55
        expected_ms_t3 = (0.9**2 + 0.2**2 + 0.5**2) / 3  # ≈ 0.367
        expected_ms_t4 = (0.1**2 + 0.9**2 + 0.5**2) / 3  # ≈ 0.357 (Winner)

        assert alignment_mean_squared._aggregate_distances(distances_t1) == pytest.approx(expected_ms_t1)
        assert alignment_mean_squared._aggregate_distances(distances_t3) == pytest.approx(expected_ms_t3)
        assert alignment_mean_squared._aggregate_distances(distances_t4) == pytest.approx(expected_ms_t4)

        # Verify template 4 has lower mean squared distance than template 3
        assert expected_ms_t4 < expected_ms_t3

    def test_reference_selection_methods_edge_case_demonstration(self):
        """Test showcasing edge case behavior differences between selection methods."""
        # Create templates for demonstration
        template1 = IrisTemplate(
            iris_codes=[np.array([[[True, False]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template2 = IrisTemplate(
            iris_codes=[np.array([[[False, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )
        template3 = IrisTemplate(
            iris_codes=[np.array([[[True, True]]])],
            mask_codes=[np.array([[[True, True]]])],
            iris_code_version="v2.1",
        )

        # Edge case: one template with very small consistent distances vs. one with mixed distances
        # Template 1 distances: [0.05, 0.05] -> sum=0.10, mean_squared=0.0025, rms=0.05
        # Template 2 distances: [0.0, 0.15]  -> sum=0.15, mean_squared=0.01125, rms=0.106
        # Template 3 distances: [0.05, 0.15] -> sum=0.20, mean_squared=0.0125, rms=0.112

        templates = [template1, template2, template3]

        # All methods should prefer template 1, but with very different confidence levels
        alignment_linear = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.LINEAR, use_first_as_reference=False
        )

        with patch("iris.nodes.templates_alignment.hamming_distance_based.simple_hamming_distance") as mock_hd:
            mock_hd.side_effect = [
                (0.05, 0),
                (0.05, 0),  # template1 distances
                (0.0, 0),
                (0.15, 0),  # template2 distances
                (0.05, 0),
                (0.15, 0),  # template3 distances
            ]

            best_idx_linear = alignment_linear._find_best_reference(templates)
            assert best_idx_linear == 0  # Template 1 wins with smallest sum

        # Verify calculations show the trade-offs
        distances_t1 = [0.05, 0.05]  # Consistent small distances
        distances_t2 = [0.0, 0.15]  # One perfect, one medium

        # LINEAR: consistent distances win (0.10 < 0.15)
        assert alignment_linear._aggregate_distances(distances_t1) == 0.10
        assert alignment_linear._aggregate_distances(distances_t2) == 0.15

        alignment_mean_squared = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.MEAN_SQUARED, use_first_as_reference=False
        )

        # MEAN_SQUARED: consistent distances still win but by smaller margin
        ms_t1 = (0.05**2 + 0.05**2) / 2  # = 0.0025
        ms_t2 = (0.0**2 + 0.15**2) / 2  # = 0.01125

        assert alignment_mean_squared._aggregate_distances(distances_t1) == pytest.approx(ms_t1)
        assert alignment_mean_squared._aggregate_distances(distances_t2) == pytest.approx(ms_t2)
        assert ms_t1 < ms_t2  # Template 1 still wins but difference is reduced

        # Show the relative differences
        linear_ratio = 0.15 / 0.10  # = 1.5 (50% worse)
        ms_ratio = ms_t2 / ms_t1  # = 4.5 (350% worse) - mean squared amplifies differences

        # Mean squared method is more sensitive to outliers
        assert ms_ratio > linear_ratio

    def test_selection_methods_sensitivity_demonstration(self):
        """Demonstrate how different methods have different sensitivity to distance outliers."""
        alignment_linear = HammingDistanceBasedAlignment(reference_selection_method=ReferenceSelectionMethod.LINEAR)
        alignment_ms = HammingDistanceBasedAlignment(reference_selection_method=ReferenceSelectionMethod.MEAN_SQUARED)
        alignment_rms = HammingDistanceBasedAlignment(
            reference_selection_method=ReferenceSelectionMethod.ROOT_MEAN_SQUARED
        )

        # Case 1: Consistent moderate distances
        consistent_distances = [0.3, 0.3, 0.3]

        # Case 2: One outlier
        outlier_distances = [0.1, 0.1, 0.7]

        # Linear method (sum)
        linear_consistent = alignment_linear._aggregate_distances(consistent_distances)  # 0.9
        linear_outlier = alignment_linear._aggregate_distances(outlier_distances)  # 0.9 (same!)

        # Mean squared method
        ms_consistent = alignment_ms._aggregate_distances(consistent_distances)  # (3 * 0.09) / 3 = 0.09
        ms_outlier = alignment_ms._aggregate_distances(outlier_distances)  # (0.01 + 0.01 + 0.49) / 3 = 0.17

        # Root mean squared
        rms_consistent = alignment_rms._aggregate_distances(consistent_distances)  # sqrt(0.09) = 0.3
        rms_outlier = alignment_rms._aggregate_distances(outlier_distances)  # sqrt(0.17) ≈ 0.412

        # Verify calculations
        assert linear_consistent == linear_outlier == pytest.approx(0.9)
        assert ms_consistent == pytest.approx(0.09)
        assert ms_outlier == pytest.approx(0.17)
        assert rms_consistent == pytest.approx(0.3)
        assert rms_outlier == pytest.approx(0.412, abs=0.001)

        # Show sensitivity differences: mean_squared and rms penalize outliers more
        assert ms_outlier > ms_consistent  # Mean squared penalizes outliers
        assert rms_outlier > rms_consistent  # RMS also penalizes outliers
        assert abs(linear_consistent - linear_outlier) < 0.001  # Linear method doesn't distinguish

    def test_integration_simple_case(self):
        """Integration test with a simple alignment case."""
        alignment = HammingDistanceBasedAlignment(rotation_shift=2)

        # Create templates where template2 is template1 shifted by 1
        iris_code1 = np.array([[[True, False, True, False]]])
        mask_code1 = np.array([[[True, True, True, True]]])

        iris_code2 = np.roll(iris_code1, 1, axis=1)  # Shifted version
        mask_code2 = mask_code1.copy()

        template1 = IrisTemplate(iris_codes=[iris_code1], mask_codes=[mask_code1], iris_code_version="v2.1")
        template2 = IrisTemplate(iris_codes=[iris_code2], mask_codes=[mask_code2], iris_code_version="v2.1")

        result = alignment.run([template1, template2])

        # First template should remain unchanged (reference)
        np.testing.assert_array_equal(result[0].iris_codes[0], iris_code1)

        # Second template should be aligned back to match the first
        # The alignment should find rotation = -1 to align template2 back to template1
        assert len(result) == 2
        assert result[0].iris_code_version == "v2.1"
        assert result[1].iris_code_version == "v2.1"
