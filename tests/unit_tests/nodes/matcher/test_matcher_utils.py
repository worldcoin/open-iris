import math
from typing import Tuple

import numpy as np
import pytest

from iris.io.dataclasses import IrisTemplate
from iris.io.errors import MatcherError
from iris.nodes.matcher.utils import hamming_distance


@pytest.mark.parametrize(
    "template_probe, template_gallery, rotation_shift, normalise, norm_mean, norm_gradient, separate_half_matching, expected_result",
    [
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            False,
            (0, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [False, True]]), np.array([[True, True], [True, False]])],
                mask_codes=[np.array([[True, False], [True, False]]), np.array([[False, True], [False, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, False]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            False,
            (0.25, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [False, False]]), np.array([[False, False], [False, False]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            True,
            (1, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            True,
            (0.8, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            0,
            False,
            None,
            None,
            False,
            (1, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            False,
            (0.22482000000000002, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [False, True]]), np.array([[True, True], [True, False]])],
                mask_codes=[np.array([[True, False], [True, False]]), np.array([[False, True], [False, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, False]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            True,
            (0.34997, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [False, False]]), np.array([[False, False], [False, False]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.000046,
            True,
            (0.7251518000000001, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            False,
            (0.6250875, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            -1,
            True,
            0.45,
            0.00005,
            False,
            (0.7251375, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            True,
            (0.6250645, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, False]]), np.array([[True, False], [False, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [True, True]]), np.array([[True, True], [True, False]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            True,
            (0.5041829166666667, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[False, True], [False, True]]), np.array([[False, False], [False, False]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, False]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            False,
            (0.475005, -1),
        ),
    ],
    ids=[
        "genuine0",
        "genuine1",
        "imposter0, half matching",
        "impostor1, half matching",
        "impostor2",
        "genuine0_norm",
        "genuine1_norm, half matching",
        "imposter0_norm, half matching",
        "impostor1_norm, half matching",
        "impostor2_norm",
        "impostor2_norm, half matching",
        "impostor3_lowerhalfnoinfo_norm, half matching",
        "impostor3_upperhalfnoinfo_norm",
    ],
)
def test_hamming_distance(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int,
    normalise: bool,
    norm_mean: float,
    norm_gradient: float,
    separate_half_matching: bool,
    expected_result: Tuple[float, ...],
) -> None:
    result = hamming_distance(
        template_probe, template_gallery, rotation_shift, normalise, norm_mean, norm_gradient, separate_half_matching
    )
    assert math.isclose(result[0], expected_result[0], rel_tol=1e-05, abs_tol=1e-05)
    assert result[1] == expected_result[1]


@pytest.mark.parametrize(
    "template_probe, template_gallery, rotation_shift, normalise, norm_mean, norm_gradient, separate_half_matching, weights, expected_result",
    [
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            None,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [False, True]]), np.array([[True, True], [True, False]])],
                mask_codes=[np.array([[True, False], [True, False]]), np.array([[False, True], [False, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, False]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            True,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.14285714285714285, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [False, False]]), np.array([[False, False], [False, False]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            False,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (1, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            False,
            None,
            None,
            False,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.8888888888888888, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            0,
            False,
            None,
            None,
            False,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (1, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            True,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.22486408163265306, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [False, True]]), np.array([[True, True], [True, False]])],
                mask_codes=[np.array([[True, False], [True, False]]), np.array([[False, True], [False, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, False]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            False,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.29636714285714283, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [False, True]]), np.array([[False, False], [False, False]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            False,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.6537342857142857, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            True,
            0.45,
            0.00005,
            False,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.6695573015873016, -1),
        ),
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, False], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[False, False], [True, False]]), np.array([[False, True], [False, False]])],
                mask_codes=[np.array([[True, False], [False, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            -1,
            True,
            0.45,
            0.00005,
            True,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.7251328571428572, 0),
        ),
        (
            IrisTemplate(
                iris_codes=[
                    np.array([[True, True], [True, True]]),
                    np.array([[True, True, True, False], [True, True, False, False]]),
                ],
                mask_codes=[
                    np.array([[True, True], [True, True]]),
                    np.array([[True, False, False, True], [True, True, True, True]]),
                ],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[
                    np.array([[False, False], [True, False]]),
                    np.array([[False, True, False, True], [False, False, True, True]]),
                ],
                mask_codes=[
                    np.array([[True, False], [False, True]]),
                    np.array([[True, True, True, True], [True, True, False, True]]),
                ],
                iris_code_version="v2.1",
            ),
            -1,
            True,
            0.45,
            0.00005,
            True,
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1, 4, 2], [1, 2, 5, 4]])],
            (0.7251492394655704, 0),
        ),
    ],
    ids=[
        "genuine0",
        "genuine1, half matching",
        "imposter0",
        "impostor1",
        "impostor2",
        "genuine0_norm, half matching",
        "genuine1_norm",
        "imposter0_norm",
        "impostor1_norm",
        "impostor2_norm, half matching",
        "impostor3_norm different size, half matching",
    ],
)
def test_hamming_distance_with_weights(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int,
    normalise: bool,
    norm_mean: float,
    norm_gradient: float,
    separate_half_matching: bool,
    weights: np.ndarray,
    expected_result: Tuple[float, ...],
) -> None:
    result = hamming_distance(
        template_probe,
        template_gallery,
        rotation_shift,
        normalise,
        norm_mean,
        norm_gradient,
        separate_half_matching,
        weights,
    )
    assert math.isclose(result[0], expected_result[0], rel_tol=1e-05, abs_tol=1e-05)
    assert result[1] == expected_result[1]


@pytest.mark.parametrize(
    "template_probe, template_gallery, rotation_shift, norm_mean, norm_gradient",
    [
        (
            IrisTemplate(
                iris_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True], [True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[np.array([[True, True]]), np.array([[True, True], [True, True]])],
                mask_codes=[np.array([[True, True]]), np.array([[True, True], [True, True]])],
                iris_code_version="v2.1",
            ),
            1,
            None,
            None,
        ),
        (
            IrisTemplate(
                iris_codes=[
                    np.array([[True, True, True], [True, True, True], [True, True, True]]),
                    np.array([[True, True, True], [True, True, True], [True, True, True]]),
                ],
                mask_codes=[
                    np.array([[True, True, False], [True, True, False], [True, True, True]]),
                    np.array([[True, True, True], [True, True, True], [True, True, True]]),
                ],
                iris_code_version="v2.1",
            ),
            IrisTemplate(
                iris_codes=[
                    np.array([[True, True, True], [True, True, False], [True, True, False]]),
                    np.array([[True, True, True], [True, True, True], [True, True, False]]),
                ],
                mask_codes=[
                    np.array([[True, True, True], [True, True, False], [True, True, False]]),
                    np.array([[True, True, True], [True, True, True], [True, True, False]]),
                ],
                iris_code_version="v2.1",
            ),
            1,
            None,
            None,
        ),
    ],
    ids=[
        "different probe and gallery size",
        "iris_code width not even",
    ],
)
def test_hamming_distance_raise_exception(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int,
    norm_mean: float,
    norm_gradient: float,
) -> None:
    with pytest.raises((MatcherError)):
        _ = hamming_distance(template_probe, template_gallery, rotation_shift, norm_mean, norm_gradient)
