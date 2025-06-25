import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.dataclasses import IrisTemplate
from iris.nodes.matcher.hamming_distance_matcher import HashBasedMatcher


@pytest.mark.parametrize(
    "rotation_shift,hash_bits",
    [
        (0, 40),
        (1, 40),
        (10, 40),
        (0, 32),
        (0, 64),
    ],
)
def test_hash_based_matcher_parameters(rotation_shift: int, hash_bits: int) -> None:
    """Test HashBasedMatcher parameter validation."""
    _ = HashBasedMatcher(rotation_shift=rotation_shift, hash_bits=hash_bits)


@pytest.mark.parametrize(
    "rotation_shift,hash_bits",
    [
        (-1, 40),  # Invalid rotation_shift
        (0, -1),  # Invalid hash_bits
        (0, 0),  # Invalid hash_bits
    ],
)
def test_hash_based_matcher_invalid_parameters(rotation_shift: int, hash_bits: int) -> None:
    """Test HashBasedMatcher invalid parameter validation."""
    with pytest.raises(ValidationError):
        _ = HashBasedMatcher(rotation_shift=rotation_shift, hash_bits=hash_bits)


def test_hash_based_matcher_unique_id_generation():
    """Test unique ID generation from iris template."""
    # Create test iris template
    iris_codes = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    mask_codes = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    template = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")

    matcher = HashBasedMatcher()
    unique_id = matcher.generate_unique_id(template)

    # Check that unique_id is an integer
    assert isinstance(unique_id, int)
    # Check that unique_id is positive
    assert unique_id >= 0
    # Check that unique_id fits in 40 bits
    assert unique_id < (1 << 40)


def test_hash_based_matcher_exact_match():
    """Test exact matching between identical templates."""
    # Create test iris template
    iris_codes = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    mask_codes = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    template1 = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")
    template2 = IrisTemplate(iris_codes=iris_codes, mask_codes=mask_codes, iris_code_version="v2.1")

    matcher = HashBasedMatcher()
    result = matcher.run(template1, template2)

    # Identical templates should match exactly (0.0)
    assert result == 0.0


def test_hash_based_matcher_no_match():
    """Test no matching between different templates."""
    # Create different test iris templates
    iris_codes1 = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    mask_codes1 = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    template1 = IrisTemplate(iris_codes=iris_codes1, mask_codes=mask_codes1, iris_code_version="v2.1")

    iris_codes2 = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    mask_codes2 = [np.random.choice([True, False], size=(16, 256, 2)) for _ in range(2)]
    template2 = IrisTemplate(iris_codes=iris_codes2, mask_codes=mask_codes2, iris_code_version="v2.1")

    matcher = HashBasedMatcher()
    result = matcher.run(template1, template2)

    # Different templates should not match (1.0)
    assert result == 1.0


def test_hash_based_matcher_id_size():
    """Test that unique ID size is correct."""
    matcher = HashBasedMatcher()
    id_size = matcher.get_id_size_bytes()

    # Should be 5 bytes for 40-bit identifier
    assert id_size == 5
