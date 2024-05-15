import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import NormalizationError
from iris.nodes.normalization.nonlinear_normalization import NonlinearNormalization
from tests.unit_tests.utils import generate_arc


@pytest.fixture
def algorithm() -> NonlinearNormalization:
    return NonlinearNormalization(
        res_in_r=2,
    )


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


def test_generate_correspondences(algorithm: NonlinearNormalization) -> None:
    pupil_points = (generate_arc(3.0, 5.0, 5.0, 0.0, 2 * np.pi, 3),)
    iris_points = (generate_arc(10.0, 4.8, 5.1, 0.0, 2 * np.pi, 3),)

    expected_correspondences = np.array(
        [
            [[10, 5], [3, 9], [3, 1]],
            [[13, 5], [1, 12], [1, -2]],
        ]
    )

    result = algorithm._generate_correspondences(
        pupil_points=pupil_points[0],
        iris_points=iris_points[0],
    )

    np.testing.assert_allclose(result, expected_correspondences, rtol=1e-05)
