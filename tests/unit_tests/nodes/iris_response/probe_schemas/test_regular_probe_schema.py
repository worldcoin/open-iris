from typing import List, Literal, Optional, Union

import pytest
from pydantic import PositiveInt, ValidationError

import iris.nodes.iris_response.probe_schemas.regular_probe_schema as rps
from iris.io.errors import ProbeSchemaError


@pytest.mark.parametrize(
    "n_rows,n_cols,boundary_rho,boundary_phi,image_shape",
    [
        (-3, 12, [0, 0], "periodic-symmetric", None),
        (6, 6, [0, 0], "periodic-symmetric", [10, 10]),
        (6, 6, [0, 0], [-1, 0], None),
        (6, 6, [0, 0], [0.6, 0.6], None),
        (6, 6, [-1, 0], 0, None),
        (6, 6, [0.6, 0.7], 0, None),
        (6, 6, [0, 0], "periodic-symmetric", [-5, 0.5]),
        (-3, 12, "some string", "some string", None),
        (5, 5, [0, 0], "periodic-symmetric", [10, 10]),
    ],
    ids=[
        "negative n_rows",
        "aliasing effects",
        "offset negative phi",
        "offset overlapping phi",
        "offset negative rho",
        "offset overlapping rho",
        "negative image_shape and float values",
        "wrong boundary option",
        "aliasing effects rho",
    ],
)
def test_regular_probe_schema_constructor_fails(
    n_rows: int,
    n_cols: int,
    boundary_rho: List[float],
    boundary_phi: Union[Literal["periodic-symmetric", "periodic-left"], List[float]],
    image_shape: Optional[List[PositiveInt]],
) -> None:
    with pytest.raises((ProbeSchemaError, ValidationError)):
        _ = rps.RegularProbeSchema(
            n_rows=n_rows, n_cols=n_cols, boundary_rho=boundary_rho, boundary_phi=boundary_phi, image_shape=image_shape
        )


@pytest.mark.parametrize(
    "n_rows,n_cols,boundary_rho,boundary_phi,image_shape",
    [
        (3, 6, [0, 0], "periodic-symmetric", None),
        (3, 5, [0, 0], "periodic-symmetric", [10, 10]),
        (3, 6, [0, 0], "periodic-left", None),
        (3, 6, [0.2, 0.5], [0.1, 0.3], None),
    ],
    ids=[
        "regular 1",
        "regular 2",
        "regular 3",
        "regular 4",
    ],
)
def test_regular_probe_schema_constructor(
    n_rows: int,
    n_cols: int,
    boundary_rho: List[float],
    boundary_phi: Union[Literal["periodic-symmetric", "periodic-left"], List[float]],
    image_shape: Optional[List[PositiveInt]],
) -> None:
    schema = rps.RegularProbeSchema(
        n_rows=n_rows, n_cols=n_cols, boundary_rho=boundary_rho, boundary_phi=boundary_phi, image_shape=image_shape
    )

    assert schema.rhos.shape == schema.phis.shape == (n_rows * n_cols,)


@pytest.mark.parametrize(
    "row_min,row_max,length,boundary_condition",
    [
        (2, 11, 10, "periodic-symmetric"),
        (2, 11, 10, "periodic-left"),
        (2, 11, 10, [0.2, 0.4]),
    ],
    ids=[
        "regular 1",
        "regular 2",
        "regular 3",
    ],
)
def test_find_suitable_n_rows(
    row_min: int,
    row_max: int,
    length: int,
    boundary_condition: Union[Literal["periodic-symmetric", "periodic-left"], List[float]],
) -> None:
    _ = rps.RegularProbeSchema.find_suitable_n_rows(
        row_min=row_min, row_max=row_max, length=length, boundary_condition=boundary_condition
    )
