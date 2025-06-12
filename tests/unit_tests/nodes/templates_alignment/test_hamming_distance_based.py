from unittest.mock import patch

import numpy as np
import pytest

from iris.io.dataclasses import IrisTemplate
from iris.nodes.templates_alignment.hamming_distance_based import HammingDistanceBasedAlignment


class TestHammingDistanceBasedAlignment:
    """Test cases for HammingDistanceBasedAlignment class."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        alignment = HammingDistanceBasedAlignment()

        assert alignment.params.rotation_shift == 15
        assert alignment.params.use_first_as_reference is True
        assert alignment.params.normalise is False

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        alignment = HammingDistanceBasedAlignment(rotation_shift=10, use_first_as_reference=False, normalise=True)

        assert alignment.params.rotation_shift == 10
        assert alignment.params.use_first_as_reference is False
        assert alignment.params.normalise is True

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
            mock_hd.assert_called_once_with(template, reference, rotation_shift=15, normalise=False)

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
