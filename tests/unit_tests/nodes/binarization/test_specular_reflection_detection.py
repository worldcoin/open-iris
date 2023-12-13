from typing import Tuple

import numpy as np
import pytest

from iris.io.dataclasses import IRImage
from iris.nodes.binarization.specular_reflection_detection import SpecularReflectionDetection


def _generate_chessboard(shape: Tuple[int, int]) -> None:
    chessboard = np.zeros(shape, dtype=np.uint8)

    chessboard[1::2, ::2] = 1
    chessboard[::2, ::2] = 1

    return chessboard


@pytest.mark.parametrize(
    "img_data,expected_result,reflection_threshold",
    [
        (np.zeros(shape=(1080, 1440), dtype=np.uint8), np.zeros(shape=(1080, 1440), dtype=np.uint8), 254),
        (np.ones(shape=(1080, 1440), dtype=np.uint8) * 255, np.ones(shape=(1080, 1440), dtype=np.uint8), 254),
        (_generate_chessboard(shape=(1080, 1440)) * 255, _generate_chessboard(shape=(1080, 1440)), 254),
        (
            np.linspace([0] * 100, [255] * 100, 100).astype(np.uint8),
            np.linspace([0] * 100, [255] * 100, 100) > 120,
            120,
        ),
    ],
    ids=["all zeros", "all 255", "chessboard test", "linear gradient image"],
)
def test_segment_specular_reflections(
    img_data: np.ndarray,
    expected_result: np.ndarray,
    reflection_threshold: int,
) -> None:
    spec_ref_algo = SpecularReflectionDetection(reflection_threshold=reflection_threshold)
    ir_image = IRImage(img_data=img_data, eye_side="right")
    result = spec_ref_algo.run(ir_image)

    np.testing.assert_equal(result.mask, expected_result)
