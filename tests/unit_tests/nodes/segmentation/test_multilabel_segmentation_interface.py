import numpy as np
import cv2
import pytest

from iris.nodes.segmentation.multilabel_segmentation_interface import MultilabelSemanticSegmentationInterface

@pytest.fixture
def multilabel_model_interface():
    return MultilabelSemanticSegmentationInterface()


@pytest.fixture
def mock_image():
    """
    h x w uint8 image:
    - left half: constant 50  (useful for thresholds >50)
    - right half: base 150 with noise (>= many thresholds; should smooth)
    """
    h, w = 96, 128
    img = np.zeros((h, w), dtype=np.uint8)
    img[:, :w//2] = 50
    rng = np.random.default_rng(0)
    right = np.full((h, w - w//2), 150, dtype=np.int16)
    noise = rng.integers(-30, 31, size=right.shape, dtype=np.int16)
    img[:, w//2:] = np.clip(right + noise, 0, 255).astype(np.uint8)
    return img

@pytest.fixture
def mock_dark_image():
    # Entire image under typical thresholds
    return np.full((64, 64), 40, dtype=np.uint8)

@pytest.fixture
def mock_bright_noisy():
    rng = np.random.default_rng(1)
    base = np.full((80, 80), 180, dtype=np.int16)
    noise = rng.integers(-35, 36, size=base.shape, dtype=np.int16)
    return np.clip(base + noise, 0, 255).astype(np.uint8)

def test_preserves_dtype_and_shape(multilabel_model_interface, mock_image):
    out = multilabel_model_interface.image_denoise(image=mock_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=75)
    assert out.shape == mock_image.shape
    assert out.dtype == mock_image.dtype == np.uint8

def test_does_not_mutate_input(multilabel_model_interface, mock_image):
    src = mock_image.copy()
    _ = multilabel_model_interface.image_denoise(image=mock_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=75)
    assert np.array_equal(mock_image, src), "Input must not be mutated"

def test_all_pixels_below_threshold_are_preserved(multilabel_model_interface, mock_dark_image):
    out = multilabel_model_interface.image_denoise(mock_dark_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=75)
    assert np.array_equal(out, mock_dark_image), "All pixels < threshold should be identical"

def test_filtered_region_is_smoothed_variance_drops(multilabel_model_interface, mock_image):
    h, w = mock_image.shape
    out = multilabel_model_interface.image_denoise(mock_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=75)
    # Right half >=75-ish — should be smoothed
    var_in = float(np.var(mock_image[:, w//2:]))
    var_out = float(np.var(out[:, w//2:]))
    assert var_out < var_in * 0.8, "Variance should drop on filtered region"

def test_mean_is_roughly_preserved_on_filtered_region(multilabel_model_interface, mock_image):
    h, w = mock_image.shape
    out = multilabel_model_interface.image_denoise(mock_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=75)
    mean_in = float(np.mean(mock_image[:, w//2:]))
    mean_out = float(np.mean(out[:, w//2:]))
    assert abs(mean_out - mean_in) <= 5.0, "Bilateral roughly preserves mean"

def test_composition_matches_manual_logic(multilabel_model_interface, mock_image):
    d, sc, ss, T = 5, 75, 10, 120
    filtered = cv2.bilateralFilter(mock_image, d=d, sigmaColor=sc, sigmaSpace=ss)
    preserve_mask = mock_image < T
    expected = filtered.copy()
    expected[preserve_mask] = mock_image[preserve_mask]
    out = multilabel_model_interface.image_denoise(mock_image, d=d, sigmaColor=sc, sigmaSpace=ss, intensityIgnore=T)
    assert np.array_equal(out, expected), "Output must equal filtered where >=T, original where <T"

def test_threshold_boundary_behavior_t_minus_1_vs_t(multilabel_model_interface):
    """
    Pixel == T-1 must be preserved exactly; pixel == T must be filtered.
    """
    T = 75
    patch = np.array(
        [[200, 200, 200],
        [200,  T-1, 200],
        [200, 200, 200]], dtype=np.uint8
    )
    patchT = patch.copy()
    patchT[1,1] = T
    out_tmin1 = multilabel_model_interface.image_denoise(patch, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=T)
    out_t     = multilabel_model_interface.image_denoise(patchT, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=T)
    assert out_tmin1[1,1] == T-1, "T-1 must be unchanged"
    assert out_t[1,1] != T, "T is not preserved (it is filtered)"

@pytest.mark.parametrize("T", [0, 1, 50, 75, 120, 254, 255])
def test_various_thresholds_respect_contract(multilabel_model_interface, mock_image, T):
    """
    - T=0: no pixel <0 ⇒ everything can be filtered (constant regions may remain identical).
    - T=255: only 255s get filtered ⇒ typical images remain identical.
    - Mid values: left (<T) preserved; right (>=T) filtered.
    """
    out = multilabel_model_interface.image_denoise(mock_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=T)
    assert out.shape == mock_image.shape and out.dtype == np.uint8
    # Check preservation of left half when 50 < T
    h, w = mock_image.shape
    left_in = mock_image[:, :w//2]
    left_out = out[:, :w//2]

    if T > 50:
        # Entire left half is < T -> preserved exactly
        assert np.array_equal(left_out, left_in)
    # Regardless of T, verify that the function never changes pixels that are strictly below T
    preserve_mask = mock_image < T
    assert np.array_equal(out[preserve_mask], mock_image[preserve_mask])
    
@pytest.mark.parametrize("d,sc,ss", [(3, 50, 10), (5, 75, 10), (9, 100, 15)])
def test_parameter_variations_keep_contract(multilabel_model_interface, mock_bright_noisy, d, sc, ss):
    """
    With a fully bright noisy image (all >= threshold),
    denoising should still reduce variance across a range of parameters.
    """
    T = 75
    out = multilabel_model_interface.image_denoise(mock_bright_noisy, d=d, sigmaColor=sc, sigmaSpace=ss, intensityIgnore=T)

    assert out.shape == mock_bright_noisy.shape
    assert out.dtype == np.uint8

    var_in = float(np.var(mock_bright_noisy))
    var_out = float(np.var(out))
    assert var_out < var_in, "Variance should drop when all pixels are eligible for filtering"


def test_extreme_thresholds(multilabel_model_interface, mock_image):
    # T=255 => effectively no filtering on typical 0..254 images
    out255 = multilabel_model_interface.image_denoise(mock_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=255)
    assert np.array_equal(out255, mock_image), "T=255 should leave typical images unchanged"

    # T=0 => allow filtering everywhere; still must not break invariants
    out0 = multilabel_model_interface.image_denoise(mock_image, d=5, sigmaColor=75, sigmaSpace=10, intensityIgnore=0)
    assert out0.shape == mock_image.shape and out0.dtype == np.uint8
    # If any region had noise, variance should not increase dramatically
    assert float(np.var(out0)) <= float(np.var(mock_image)) * 1.2