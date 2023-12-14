import numpy as np
import pytest
from pydantic import ValidationError

from iris.io.errors import NormalizationError
from iris.nodes.normalization.perspective_normalization import PerspectiveNormalization


@pytest.fixture
def algorithm() -> PerspectiveNormalization:
    return PerspectiveNormalization(
        res_in_phi=400,
        res_in_r=100,
        skip_boundary_points=10,
        intermediate_radiuses=np.linspace(0.0, 1.0, 10),
    )


@pytest.mark.parametrize(
    "wrong_param",
    [
        ({"res_in_phi": 0, "res_in_r": 10}),
        ({"res_in_phi": -1, "res_in_r": 10}),
        ({"res_in_phi": 10, "res_in_r": 0}),
        ({"res_in_phi": 10, "res_in_r": -1}),
        ({"skip_boundary_points": -1}),
        ({"intermediate_radiuses": []}),
        ({"intermediate_radiuses": [0.0]}),
        ({"intermediate_radiuses": [-1.0, 0.0, 1.0]}),
        ({"intermediate_radiuses": [0, 0.2, 1.2]}),
    ],
)
def test_constructor_raises_exception(wrong_param: dict) -> None:
    with pytest.raises((NormalizationError, ValidationError)):
        _ = PerspectiveNormalization(**wrong_param)


def test_bbox_coords(algorithm: PerspectiveNormalization) -> None:
    # fmt: off
    norm_dst_points = np.array([
        [0, 0],
        [11, 0],
        [0, 11],
        [11, 11],
    ])
    # fmt: on

    expected_result = (0, 0, 11, 11)

    result = algorithm._bbox_coords(norm_dst_points)

    assert result == expected_result


def test_correspondence_rois_coords(algorithm: PerspectiveNormalization) -> None:
    angle_idx = 0
    ring_idx = 0
    # fmt: off
    # Nones shouldn't be taken
    src_points = np.array([
        [[0.0, 1.0], [1.0, 2.0], [None, None]],
        [[1.0, 1.0], [2.0, 1.0], [None, None]],
        [[None, None], [None, None], [None, None]],
    ])
    dst_points = np.array([
        [[11.0, 11.0], [12.0, 21.0], [None, None]],
        [[22.0, 21.0], [22.0, 22.0], [None, None]],
        [[None, None], [None, None], [None, None]],
    ])

    expected_src_roi = np.array([
        [0.0, 1.0], [1.0, 2.0], [1.0, 1.0], [2.0, 1.0]
    ])
    expected_dst_roi = np.array([
        [11.0, 11.0], [12.0, 21.0], [22.0, 21.0], [22.0, 22.0],
    ])
    # fmt: on

    result_src_roi, result_dst_roi = algorithm._correspondence_rois_coords(
        angle_idx=angle_idx,
        ring_idx=ring_idx,
        src_points=src_points,
        dst_points=dst_points,
    )

    assert np.all(result_src_roi == expected_src_roi)
    assert np.all(result_dst_roi == expected_dst_roi)


def test_cartesian2homogeneous() -> None:
    cartesian_xs = np.array([1.0, 2.0, 3.0])
    cartesian_ys = np.array([10.0, 20.0, 30.0])

    cartesian_pts = np.array([cartesian_xs, cartesian_ys])
    # fmt: off
    expected_homogeneous_pts = np.array([
        [1.0, 2.0, 3.0],        # xs
        [10.0, 20.0, 30.0],     # ys
        [1.0, 1.0, 1.0]         # ks
    ])
    # fmt: on

    result = PerspectiveNormalization.cartesian2homogeneous(cartesian_pts)

    assert np.all(result == expected_homogeneous_pts)


def test_homogeneous2cartesian() -> None:
    # fmt: off
    homogeneous_pts = np.array([
        [1.0, 2.0, 3.0],        # xs
        [10.0, 20.0, 30.0],     # ys
        [1.0, 2.0, 3.0]         # ks
    ])
    expected_cartesian_pts = np.array([
        [1.0, 1.0, 1.0],        # xs
        [10.0, 10.0, 10.0],     # ys
    ])
    # fmt: on

    result = PerspectiveNormalization.homogeneous2cartesian(homogeneous_pts)

    assert np.all(result == expected_cartesian_pts)
