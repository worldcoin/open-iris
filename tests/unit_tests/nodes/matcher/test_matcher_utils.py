import math
from typing import Tuple

import numpy as np
import pytest

from iris.io.dataclasses import IrisTemplate
from iris.io.errors import MatcherError
from iris.nodes.matcher.utils import hamming_distance


@pytest.mark.parametrize(
    "template_probe, template_gallery, rotation_shift, normalise, nm_dist, nm_type, expected_result",
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
            "sqrt",
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
            True,
            0.45,
            "sqrt",
            (0.2867006838144548, -1),
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
            "sqrt",
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
            True,
            0.45,
            "sqrt",
            (0.7703179042939132, -1),
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
            "linear",
            (0.7645670365077234, -1),
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
            "sqrt",
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
            True,
            0.45,
            "sqrt",
            (0.7703179042939132, -1),
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
            "linear",
            (0.7645670365077234, -1),
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
            "sqrt",
            (0.6349365679618759, 0),
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
            "sqrt",
            (0.48484617125293383, -1),
        ),
    ],
    ids=[
        "genuine0",
        "genuine1",
        "imposter0",
        "impostor1",
        "impostor2",
        "genuine0_norm",
        "genuine1_norm",
        "imposter0_norm",
        "impostor1_norm with sqrt nm_type ",
        "impostor1_norm with linear nm_type ",
        "impostor2_norm",
        "impostor3_norm with sqrt type",
        "impostor3_norm with linear type",
        "impostor4_lowerhalfnoinfo_norm",
        "impostor5_upperhalfnoinfo_norm",
    ],
)
def test_hamming_distance(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int,
    normalise: bool,
    nm_dist: float,
    nm_type: str,
    expected_result: Tuple[float, ...],
) -> None:
    result = hamming_distance(template_probe, template_gallery, rotation_shift, normalise, nm_dist, nm_type)
    assert math.isclose(result[0], expected_result[0], rel_tol=1e-05, abs_tol=1e-05)
    assert result[1] == expected_result[1]


@pytest.mark.parametrize(
    "template_probe, template_gallery, rotation_shift, normalise, nm_dist, nm_type, weights, expected_result",
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
            "sqrt",
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
            True,
            0.45,
            "sqrt",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.22967674843904062, -1),
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
            "linear",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.23281720292127467, -1),
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
            "sqrt",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.8512211593818028, 0),
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
            "linear",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.8571428571428571, 0),
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
            "sqrt",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.8020834362167217, -1),
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
            "linear",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (0.8011224867405204, -1),
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
            "sqrt",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1], [1, 2]])],
            (1, 0),
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
            "sqrt",
            [np.array([[3, 1], [1, 2]]), np.array([[3, 1, 4, 2], [1, 2, 5, 4]])],
            (1, 0),
        ),
    ],
    ids=[
        "genuine0",
        "genuine1",
        "imposter0",
        "impostor1",
        "impostor2",
        "genuine0_norm",
        "genuine1_norm",
        "imposter0_norm with nm_type sqrt",
        "imposter0_norm with nm_type linear",
        "impostor1_norm with nm_type sqrt",
        "impostor1_norm with nm_type linear",
        "impostor2_norm with nm_type sqrt",
        "impostor2_norm with nm_type linear",
        "impostor3_norm different size",
    ],
)
def test_hamming_distance_with_weights(
    template_probe: IrisTemplate,
    template_gallery: IrisTemplate,
    rotation_shift: int,
    normalise: bool,
    nm_dist: float,
    nm_type: str,
    weights: np.ndarray,
    expected_result: Tuple[float, ...],
) -> None:
    result = hamming_distance(template_probe, template_gallery, rotation_shift, normalise, nm_dist, nm_type, weights)

    assert math.isclose(result[0], expected_result[0], rel_tol=1e-05, abs_tol=1e-05)
    assert result[1] == expected_result[1]


@pytest.mark.parametrize(
    "template_probe, template_gallery, rotation_shift, nm_dist",
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
    nm_dist: float,
) -> None:
    with pytest.raises((MatcherError)):
        _ = hamming_distance(template_probe, template_gallery, rotation_shift, nm_dist)
