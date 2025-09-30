import numpy as np
import pytest

import iris.io.errors as E
from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisTemplate, IrisTemplateWithId
from iris.nodes.validators.object_validators import AreTemplatesAggregationCompatible


@pytest.fixture
def valid_template():
    """Create a valid IrisTemplate for testing."""
    iris_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
    mask_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
    return IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")


@pytest.fixture
def compatible_templates():
    """Create a list of compatible IrisTemplates for testing."""
    templates = []
    for i in range(3):
        iris_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
        mask_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
        template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
        templates.append(template)
    return templates


@pytest.fixture
def compatible_templates_with_id(compatible_templates):
    """Create a list of compatible IrisTemplatesWithId for testing."""
    return [
        IrisTemplateWithId.from_template(template, f"image_id_{i}") for i, template in enumerate(compatible_templates)
    ]


@pytest.fixture
def simple_compatible_templates():
    """Create simple compatible templates with known structure."""
    iris_codes_1 = [np.array([[[True, False], [False, True]]]).astype(bool)]
    mask_codes_1 = [np.array([[[True, True], [True, True]]]).astype(bool)]
    template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

    iris_codes_2 = [np.array([[[False, True], [True, False]]]).astype(bool)]
    mask_codes_2 = [np.array([[[True, False], [False, True]]]).astype(bool)]
    template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

    return [template1, template2]


class TestAreTemplatesAggregationCompatible:
    """Test suite for AreTemplatesAggregationCompatible validator class."""

    def test_init_default(self):
        """Test initialization with default parameters."""
        validator = AreTemplatesAggregationCompatible()

        # Should inherit from both Callback and Algorithm
        assert isinstance(validator, Callback)
        assert isinstance(validator, Algorithm)

    def test_run_empty_templates_list(self):
        """Test run method with empty templates list."""
        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(E.TemplateAggregationCompatibilityError, match="No templates provided for validation"):
            validator.run([])

    def test_run_single_template(self, valid_template):
        """Test run method with single template (should pass)."""
        validator = AreTemplatesAggregationCompatible()

        # Single template should always be compatible
        try:
            validator.run([valid_template])
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("Single template should always be compatible")

    def test_run_compatible_templates(self, compatible_templates):
        """Test run method with compatible templates."""
        validator = AreTemplatesAggregationCompatible()

        # Compatible templates should not raise an exception
        try:
            validator.run(compatible_templates)
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("Compatible templates should not raise an exception")

    def test_run_simple_compatible_templates(self, simple_compatible_templates):
        """Test run method with simple compatible templates."""
        validator = AreTemplatesAggregationCompatible()

        # Should not raise an exception
        try:
            validator.run(simple_compatible_templates)
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("Simple compatible templates should not raise an exception")

    def test_run_different_iris_code_versions(self):
        """Test run method with templates having different iris code versions."""
        iris_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
        mask_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]

        template1 = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
        template2 = IrisTemplate(
            iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.2"
        )  # Different version

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(
            E.TemplateAggregationCompatibilityError, match="Templates have different iris code versions"
        ):
            validator.run([template1, template2])

    def test_run_different_number_of_wavelets(self):
        """Test run method with templates having different numbers of wavelets."""
        # Template 1 with 2 wavelets
        iris_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
        mask_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        # Template 2 with 3 wavelets
        iris_codes_2 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(3)]
        mask_codes_2 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(3)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(
            E.TemplateAggregationCompatibilityError, match="Templates have different numbers of wavelets"
        ):
            validator.run([template1, template2])

    def test_run_different_iris_code_shapes(self):
        """Test run method with templates having different iris code shapes."""
        # Template 1 with shape (16, 256, 2)
        iris_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]
        mask_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        # Template 2 with different shape (32, 128, 2)
        iris_codes_2 = [np.random.choice(2, size=(32, 128, 2)).astype(bool)]
        mask_codes_2 = [np.random.choice(2, size=(32, 128, 2)).astype(bool)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(
            E.TemplateAggregationCompatibilityError, match="Iris codes for wavelet 0 have different shapes"
        ):
            validator.run([template1, template2])

    def test_run_multiple_wavelets_different_shapes(self):
        """Test run method with multiple wavelets where one has different shapes."""
        # Template 1 with 2 wavelets, both same shape
        iris_codes_1 = [
            np.random.choice(2, size=(16, 256, 2)).astype(bool),
            np.random.choice(2, size=(16, 256, 2)).astype(bool),
        ]
        mask_codes_1 = [
            np.random.choice(2, size=(16, 256, 2)).astype(bool),
            np.random.choice(2, size=(16, 256, 2)).astype(bool),
        ]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        # Template 2 with 2 wavelets, second wavelet has different shape
        iris_codes_2 = [
            np.random.choice(2, size=(16, 256, 2)).astype(bool),  # Same as template1
            np.random.choice(2, size=(32, 128, 2)).astype(bool),  # Different shape
        ]
        mask_codes_2 = [
            np.random.choice(2, size=(16, 256, 2)).astype(bool),  # Same as template1
            np.random.choice(2, size=(32, 128, 2)).astype(bool),  # Different shape
        ]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(
            E.TemplateAggregationCompatibilityError, match="Iris codes for wavelet 1 have different shapes"
        ):
            validator.run([template1, template2])

    def test_on_execute_start_method(self, compatible_templates_with_id):
        """Test on_execute_start method (callback interface)."""
        validator = AreTemplatesAggregationCompatible()

        # Should call the run method internally
        try:
            validator.on_execute_start(compatible_templates_with_id)
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("on_execute_start should not raise exception for compatible templates")

    def test_on_execute_start_with_incompatible_templates(self):
        """Test on_execute_start method with incompatible templates."""
        iris_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]
        mask_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]

        template1 = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
        template1_with_id = IrisTemplateWithId.from_template(template1, "image_id_0")
        template2 = IrisTemplate(
            iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.2"
        )  # Different version
        template2_with_id = IrisTemplateWithId.from_template(template2, "image_id_1")

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(E.TemplateAggregationCompatibilityError):
            validator.on_execute_start([template1_with_id, template2_with_id])

    def test_on_execute_start_with_additional_args(self, compatible_templates_with_id):
        """Test on_execute_start method with additional args and kwargs."""
        validator = AreTemplatesAggregationCompatible()

        # Should work with additional arguments (they should be ignored)
        try:
            validator.on_execute_start(compatible_templates_with_id, "extra_arg", extra_kwarg="value")
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("on_execute_start should not raise exception for compatible templates")

    def test_inheritance_from_callback_and_algorithm(self):
        """Test that the class properly inherits from both Callback and Algorithm."""
        validator = AreTemplatesAggregationCompatible()

        # Should be instance of both
        assert isinstance(validator, Callback)
        assert isinstance(validator, Algorithm)

        # Should have methods from both interfaces
        assert hasattr(validator, "on_execute_start")  # From Callback
        assert hasattr(validator, "run")  # From Algorithm
        assert hasattr(validator, "execute")  # From Algorithm

    def test_algorithm_interface_methods(self, compatible_templates):
        """Test that Algorithm interface methods work correctly."""
        validator = AreTemplatesAggregationCompatible()

        # Test execute method (should call run internally)
        try:
            validator.execute(compatible_templates)
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("execute should not raise exception for compatible templates")

        # Test callable interface
        try:
            validator(compatible_templates)
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("callable interface should not raise exception for compatible templates")

    def test_edge_case_two_identical_templates(self, valid_template):
        """Test with two identical templates."""
        validator = AreTemplatesAggregationCompatible()

        # Two identical templates should be compatible
        try:
            validator.run([valid_template, valid_template])
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("Identical templates should be compatible")

    def test_edge_case_many_compatible_templates(self):
        """Test with many compatible templates."""
        templates = []
        for i in range(10):  # Create 10 compatible templates
            iris_codes = [np.random.choice(2, size=(8, 64, 2)).astype(bool) for _ in range(2)]
            mask_codes = [np.random.choice(2, size=(8, 64, 2)).astype(bool) for _ in range(2)]
            template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
            templates.append(template)

        validator = AreTemplatesAggregationCompatible()

        try:
            validator.run(templates)
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("Many compatible templates should not raise exception")

    def test_edge_case_minimal_templates(self):
        """Test with minimal size templates."""
        # Create very small templates
        iris_codes_1 = [np.array([[[True, False]]]).astype(bool)]
        mask_codes_1 = [np.array([[[True, True]]]).astype(bool)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        iris_codes_2 = [np.array([[[False, True]]]).astype(bool)]
        mask_codes_2 = [np.array([[[False, False]]]).astype(bool)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        validator = AreTemplatesAggregationCompatible()

        try:
            validator.run([template1, template2])
        except E.TemplateAggregationCompatibilityError:
            pytest.fail("Minimal compatible templates should not raise exception")

    def test_error_message_content_iris_code_version(self):
        """Test that error messages contain appropriate information."""
        iris_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]
        mask_codes = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]

        template1 = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
        template2 = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v3.0")

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(E.TemplateAggregationCompatibilityError) as exc_info:
            validator.run([template1, template2])

        assert "different iris code versions" in str(exc_info.value)

    def test_error_message_content_wavelet_count(self):
        """Test error message for different wavelet counts."""
        iris_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
        mask_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(2)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        iris_codes_2 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(3)]
        mask_codes_2 = [np.random.choice(2, size=(16, 256, 2)).astype(bool) for _ in range(3)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(E.TemplateAggregationCompatibilityError) as exc_info:
            validator.run([template1, template2])

        assert "different numbers of wavelets" in str(exc_info.value)

    def test_error_message_content_iris_shape(self):
        """Test error message for different iris code shapes."""
        iris_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]
        mask_codes_1 = [np.random.choice(2, size=(16, 256, 2)).astype(bool)]
        template1 = IrisTemplate(iris_codes=iris_codes_1, mask_codes=mask_codes_1, iris_code_version="v2.1")

        iris_codes_2 = [np.random.choice(2, size=(32, 128, 2)).astype(bool)]
        mask_codes_2 = [np.random.choice(2, size=(32, 128, 2)).astype(bool)]
        template2 = IrisTemplate(iris_codes=iris_codes_2, mask_codes=mask_codes_2, iris_code_version="v2.1")

        validator = AreTemplatesAggregationCompatible()

        with pytest.raises(E.TemplateAggregationCompatibilityError) as exc_info:
            validator.run([template1, template2])

        assert "Iris codes for wavelet 0 have different shapes" in str(exc_info.value)
