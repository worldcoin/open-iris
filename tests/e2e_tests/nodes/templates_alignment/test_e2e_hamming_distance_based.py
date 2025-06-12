import numpy as np
import pytest

from iris.io.dataclasses import IrisTemplate
from iris.nodes.templates_alignment.hamming_distance_based import HammingDistanceBasedAlignment


class TestE2EHammingDistanceBasedAlignment:
    """End-to-end test cases for HammingDistanceBasedAlignment class."""

    @pytest.fixture
    def sample_templates(self):
        """Create sample iris templates for testing."""
        # Create realistic iris code dimensions (16, 256, 2)
        template1 = IrisTemplate(
            iris_codes=[
                np.random.choice([True, False], size=(16, 256, 2)),
                np.random.choice([True, False], size=(16, 256, 2)),
            ],
            mask_codes=[
                np.random.choice([True, False], size=(16, 256, 2), p=[0.8, 0.2]),
                np.random.choice([True, False], size=(16, 256, 2), p=[0.8, 0.2]),
            ],
            iris_code_version="v2.1",
        )

        # Create second template by rotating first template
        template2 = IrisTemplate(
            iris_codes=[np.roll(template1.iris_codes[0], 5, axis=1), np.roll(template1.iris_codes[1], 5, axis=1)],
            mask_codes=[np.roll(template1.mask_codes[0], 5, axis=1), np.roll(template1.mask_codes[1], 5, axis=1)],
            iris_code_version="v2.1",
        )

        # Create third template with different pattern
        template3 = IrisTemplate(
            iris_codes=[
                np.random.choice([True, False], size=(16, 256, 2)),
                np.random.choice([True, False], size=(16, 256, 2)),
            ],
            mask_codes=[
                np.random.choice([True, False], size=(16, 256, 2), p=[0.7, 0.3]),
                np.random.choice([True, False], size=(16, 256, 2), p=[0.7, 0.3]),
            ],
            iris_code_version="v2.1",
        )

        return [template1, template2, template3]

    def test_e2e_alignment_default_parameters(self, sample_templates):
        """Test end-to-end alignment with default parameters."""
        alignment = HammingDistanceBasedAlignment()

        result = alignment.run(sample_templates)

        # Check that we get the same number of templates back
        assert len(result) == len(sample_templates)

        # Check that all templates have the same structure
        for template in result:
            assert len(template.iris_codes) == 2
            assert len(template.mask_codes) == 2
            assert template.iris_code_version == "v2.1"

            for iris_code, mask_code in zip(template.iris_codes, template.mask_codes):
                assert iris_code.shape == (16, 256, 2)
                assert mask_code.shape == (16, 256, 2)

    def test_e2e_alignment_first_as_reference(self, sample_templates):
        """Test end-to-end alignment using first template as reference."""
        alignment = HammingDistanceBasedAlignment(use_first_as_reference=True, rotation_shift=10)

        result = alignment.run(sample_templates)

        # First template should remain unchanged
        for i in range(len(sample_templates[0].iris_codes)):
            np.testing.assert_array_equal(result[0].iris_codes[i], sample_templates[0].iris_codes[i])
            np.testing.assert_array_equal(result[0].mask_codes[i], sample_templates[0].mask_codes[i])

    def test_e2e_alignment_best_reference(self, sample_templates):
        """Test end-to-end alignment using best template as reference."""
        alignment = HammingDistanceBasedAlignment(use_first_as_reference=False, rotation_shift=15)

        result = alignment.run(sample_templates)

        # Check that alignment completed successfully
        assert len(result) == len(sample_templates)

        # All templates should have consistent structure
        for template in result:
            assert len(template.iris_codes) == len(sample_templates[0].iris_codes)
            assert len(template.mask_codes) == len(sample_templates[0].mask_codes)

    def test_e2e_known_rotation_alignment(self):
        """Test alignment with known rotation offset."""
        # Create base template
        base_iris = np.random.choice([True, False], size=(16, 64, 2))  # Smaller for easier testing
        base_mask = np.ones((16, 64, 2), dtype=bool)  # All valid

        base_template = IrisTemplate(iris_codes=[base_iris], mask_codes=[base_mask], iris_code_version="v2.1")

        # Create rotated version
        rotation_offset = 3
        rotated_template = IrisTemplate(
            iris_codes=[np.roll(base_iris, rotation_offset, axis=1)],
            mask_codes=[np.roll(base_mask, rotation_offset, axis=1)],
            iris_code_version="v2.1",
        )

        alignment = HammingDistanceBasedAlignment(rotation_shift=10)
        result = alignment.run([base_template, rotated_template])

        # First template (reference) should be unchanged
        np.testing.assert_array_equal(result[0].iris_codes[0], base_template.iris_codes[0])

        # Second template should be aligned to minimize hamming distance
        assert len(result) == 2
        assert result[1].iris_code_version == "v2.1"

    def test_e2e_single_template(self):
        """Test with single template."""
        template = IrisTemplate(
            iris_codes=[np.random.choice([True, False], size=(16, 256, 2))],
            mask_codes=[np.ones((16, 256, 2), dtype=bool)],
            iris_code_version="v2.1",
        )

        alignment = HammingDistanceBasedAlignment()
        result = alignment.run([template])

        assert len(result) == 1
        np.testing.assert_array_equal(result[0].iris_codes[0], template.iris_codes[0])
        np.testing.assert_array_equal(result[0].mask_codes[0], template.mask_codes[0])

    def test_e2e_empty_templates_raises_error(self):
        """Test that empty template list raises ValueError."""
        alignment = HammingDistanceBasedAlignment()

        with pytest.raises(ValueError, match="No templates provided for alignment"):
            alignment.run([])

    def test_e2e_different_template_structures(self):
        """Test with templates having different internal structures."""
        # Template with 1 wavelet
        template1 = IrisTemplate(
            iris_codes=[np.random.choice([True, False], size=(16, 256, 2))],
            mask_codes=[np.ones((16, 256, 2), dtype=bool)],
            iris_code_version="v2.1",
        )

        # Template with 2 wavelets
        template2 = IrisTemplate(
            iris_codes=[
                np.random.choice([True, False], size=(16, 256, 2)),
                np.random.choice([True, False], size=(16, 256, 2)),
            ],
            mask_codes=[np.ones((16, 256, 2), dtype=bool), np.ones((16, 256, 2), dtype=bool)],
            iris_code_version="v2.1",
        )

        alignment = HammingDistanceBasedAlignment()

        # This should work since hamming distance matcher should handle different structures
        # or raise appropriate errors if incompatible
        try:
            result = alignment.run([template1, template2])
            # If it succeeds, check basic properties
            assert len(result) == 2
        except Exception as e:
            # If it fails, it should be due to incompatible template structures
            assert "different sizes" in str(e) or "MatcherError" in str(type(e).__name__)

    @pytest.mark.parametrize("rotation_shift", [5, 10, 15, 20])
    def test_e2e_different_rotation_shifts(self, rotation_shift):
        """Test alignment with different rotation shift parameters."""
        # Create simple templates
        base_template = IrisTemplate(
            iris_codes=[np.random.choice([True, False], size=(8, 32, 2))],
            mask_codes=[np.ones((8, 32, 2), dtype=bool)],
            iris_code_version="v2.1",
        )

        rotated_template = IrisTemplate(
            iris_codes=[np.roll(base_template.iris_codes[0], 2, axis=1)],
            mask_codes=[np.roll(base_template.mask_codes[0], 2, axis=1)],
            iris_code_version="v2.1",
        )

        alignment = HammingDistanceBasedAlignment(rotation_shift=rotation_shift)
        result = alignment.run([base_template, rotated_template])

        assert len(result) == 2
        # Both templates should maintain their basic structure
        for template in result:
            assert template.iris_codes[0].shape == (8, 32, 2)
            assert template.mask_codes[0].shape == (8, 32, 2)

    @pytest.mark.parametrize("normalise", [True, False])
    def test_e2e_normalise_parameter(self, normalise):
        """Test alignment with different normalise parameter values."""
        # Create simple templates
        base_template = IrisTemplate(
            iris_codes=[np.random.choice([True, False], size=(8, 32, 2))],
            mask_codes=[np.ones((8, 32, 2), dtype=bool)],
            iris_code_version="v2.1",
        )

        rotated_template = IrisTemplate(
            iris_codes=[np.roll(base_template.iris_codes[0], 3, axis=1)],
            mask_codes=[np.roll(base_template.mask_codes[0], 3, axis=1)],
            iris_code_version="v2.1",
        )

        alignment = HammingDistanceBasedAlignment(normalise=normalise, rotation_shift=10)
        result = alignment.run([base_template, rotated_template])

        assert len(result) == 2
        # Both templates should maintain their basic structure
        for template in result:
            assert template.iris_codes[0].shape == (8, 32, 2)
            assert template.mask_codes[0].shape == (8, 32, 2)
            assert template.iris_code_version == "v2.1"

        # Verify that alignment parameter is correctly set
        assert alignment.params.normalise == normalise
