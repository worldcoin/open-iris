import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import NormalizationError
from iris.nodes.normalization.linear_normalization import LinearNormalization
from tests.unit_tests.utils import generate_arc


@pytest.mark.parametrize(
    "wrong_param",
    [
        ({"res_in_r": -1}),
        ({"res_in_r": 0}),
    ],
)
def test_constructor_raises_exception(wrong_param: dict) -> None:
    with pytest.raises((NormalizationError, ValidationError)):
        _ = LinearNormalization(**wrong_param)


@pytest.mark.parametrize(
    "pupil_points, iris_points, expected_correspondences",
    [
        (
            generate_arc(3.0, 5.0, 5.0, 0.0, 2 * np.pi, 3),
            generate_arc(10.0, 4.8, 5.1, 0.0, 2 * np.pi, 3),
            np.array(
                [
                    [[8, 5], [4, 8], [3, 2]],
                    [[15, 5], [0, 14], [0, -4]],
                ]
            ),
        ),
        (
            generate_arc(50.0, 0.0, 0.0, 0.0, 2 * np.pi, 8),
            generate_arc(100.0, 0.0, 0.0, 0.0, 2 * np.pi, 8),
            np.array(
                [
                    [[50, 0], [35, 35], [0, 50], [-35, 35], [-50, 0], [-35, -35], [0, -50], [35, -35]],
                    [[100, 0], [71, 71], [0, 100], [-71, 71], [-100, 0], [-71, -71], [0, -100], [71, -71]],
                ]
            ),
        ),
    ],
    ids=[
        "test1",
        "test2",
    ],
)
def test_generate_correspondences(
    pupil_points: np.ndarray, iris_points: np.ndarray, expected_correspondences: np.ndarray
) -> None:
    algorithm = LinearNormalization(
        res_in_r=2,
    )
    result = algorithm._generate_correspondences(
        pupil_points=pupil_points,
        iris_points=iris_points,
    )

    np.testing.assert_allclose(result, expected_correspondences, rtol=1e-05)
