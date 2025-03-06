import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import NormalizationError
from iris.nodes.normalization.nonlinear_normalization import NonlinearNormalization, NonlinearType
from tests.unit_tests.utils import generate_arc


@pytest.mark.parametrize(
    "wrong_param",
    [
        ({"res_in_r": -10}),
        ({"res_in_r": 0}),
    ],
)
def test_constructor_raises_exception(wrong_param: dict) -> None:
    with pytest.raises((NormalizationError, ValidationError)):
        _ = NonlinearNormalization(**wrong_param)


@pytest.mark.parametrize(
    "res_in_r, method, expected_correspondences",
    [
        (
            2,
            NonlinearType.default,
            np.array(
                [
                    [[10, 5], [3, 9], [3, 1]],
                    [[13, 5], [1, 12], [1, -2]],
                ]
            ),
        ),
        (
            2,
            NonlinearType.wyatt,
            np.array(
                [
                    [[11, 5], [2, 11], [2, 0]],
                    [[15, 5], [0, 14], [0, -4]],
                ]
            ),
        ),
    ],
    ids=[
        "test1",
        "test2",
    ],
)
def test_generate_correspondences(res_in_r: int, method: NonlinearType, expected_correspondences: np.array) -> None:
    algorithm = NonlinearNormalization(
        res_in_r=res_in_r,
        method=method,
    )
    pupil_points = (generate_arc(3.0, 5.0, 5.0, 0.0, 2 * np.pi, 3),)
    iris_points = (generate_arc(10.0, 4.8, 5.1, 0.0, 2 * np.pi, 3),)

    result = algorithm._generate_correspondences(
        pupil_points=pupil_points[0], iris_points=iris_points[0], p2i_ratio=0.3
    )

    np.testing.assert_allclose(result, expected_correspondences, rtol=1e-05)
