from typing import List, Literal, Optional, Tuple, Union

import numpy as np
from pydantic import Field, PositiveInt, confloat, fields, validator

from iris.io.errors import ProbeSchemaError
from iris.nodes.iris_response.probe_schemas.probe_schema_interface import ProbeSchema


class RegularProbeSchema(ProbeSchema):
    """Probe Schema for a regular Grid."""

    class RegularProbeSchemaParameters(ProbeSchema.ProbeSchemaParameters):
        """RegularProbeSchema parameters."""

        n_rows: int = Field(..., gt=1)
        n_cols: int = Field(..., gt=1)
        boundary_rho: List[confloat(ge=0.0, lt=1)]
        boundary_phi: Union[
            Literal["periodic-symmetric", "periodic-left"],
            List[confloat(ge=0.0, lt=1)],
        ]
        image_shape: Optional[List[PositiveInt]]

        @validator("boundary_rho", "boundary_phi")
        def check_overlap(
            cls: type,
            v: Union[Literal["periodic-symmetric", "periodic-left"], List[confloat(ge=0.0, lt=1)]],
            field: fields.ModelField,
        ) -> Union[Literal["periodic-symmetric", "periodic-left"], List[confloat(ge=0.0, lt=1)]]:
            """Validate offsets to avoid overlap.

            Args:
                cls (type): Class type.
                v (Union[Literal["periodic-symmetric", "periodic-left"], List[confloat(ge=0.0, lt=1)]]): Value to check.
                field (fields.ModelField): Field descriptor.

            Raises:
                ProbeSchemaError: Raises warning that offsets are together too large.

            Returns:
                Union[Literal["periodic-symmetric", "periodic-left"], List[confloat(ge=0.0, lt=1)]]: The value for boundary_rho or boundary_phi respectively
            """
            if isinstance(v, List):
                if (v[0] + v[1]) >= 1:
                    raise ProbeSchemaError(
                        f"Offset for {field.name} on left and right corner must be a sum smaller 1, otherwise, offsets overlap."
                    )

            return v

    __parameters_type__ = RegularProbeSchemaParameters

    def __init__(
        self,
        n_rows: int,
        n_cols: int,
        boundary_rho: List[float] = [0, 0.0625],
        boundary_phi: Union[
            Literal["periodic-symmetric", "periodic-left"], List[confloat(ge=0.0, lt=1)]
        ] = "periodic-left",
        image_shape: Optional[List[PositiveInt]] = None,
    ) -> None:
        """Assign parameters.

        Args:
            n_rows (int): Number of rows used, represents the number of different rho
                        values
            n_cols (int): Number of columns used, represents the number of different
                            phi values
            boundary_rho (List[float], optional): List with two values f1 and f2. The sampling goes from 0+f1 to 0-f2.
            boundary_phi (Union[Literal["periodic-symmetric", "periodic-left"], List[confloat(ge=0.0, lt=1)]], optional): Boundary conditions for the probing
                can either be periodic or non-periodic, if they are periodic, the distance
                from one column to the next must be the same also for the boundaries.
                Else, no conditions for the boundaries are required. Options are:
                    - 'periodic-symmetric': the first and the last column are placed with an offset to the
                                borders, that is half of the spacing of the two columns
                    - 'periodic-left': the first column is at the border of the bottom of the image, while
                            the last column is one spacing apart from the top of the image
                    - list with two values: in this case the an offset of value f1 and f2 is set on both ends, i.e. the
                            the sampling no longer goes from 0 to 1 ('no-offset') but instead from 0+f1 to 0-f2
                Defaults to "periodic_symmetric".
            image_shape (list, optional): list containing the desired image dimensions. If provided, the function will throw
                a warning if interpolation happens, i.e. if a kernel would be placed in between two pixels. Defaults to None.
        """
        super().__init__(
            n_rows=n_rows,
            n_cols=n_cols,
            boundary_rho=boundary_rho,
            boundary_phi=boundary_phi,
            image_shape=image_shape,
        )

    def generate_schema(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rhos and phis.

        Return:
            Tuple[np.ndarray, np.ndarray]: the rhos and phis.
        """
        rho = np.linspace(
            0 + self.params.boundary_rho[0], 1 - self.params.boundary_rho[1], self.params.n_rows, endpoint=True
        )

        if self.params.boundary_phi == "periodic-symmetric":
            phi = np.linspace(0, 1, self.params.n_cols, endpoint=False)
            phi = phi + (phi[1] - phi[0]) / 2

        if self.params.boundary_phi == "periodic-left":
            phi = np.linspace(0, 1, self.params.n_cols, endpoint=False)

        if isinstance(self.params.boundary_phi, List):
            phi = np.linspace(
                0 + self.params.boundary_phi[0], 1 - self.params.boundary_phi[1], self.params.n_cols, endpoint=True
            )
        phis, rhos = np.meshgrid(phi, rho)
        rhos = rhos.flatten()
        phis = phis.flatten()

        # if image_shape provided: verify that values lie on pixel values
        if self.params.image_shape is not None:
            rhos_pixel_values = rhos * self.params.image_shape[0]
            phis_pixel_values = phis * self.params.image_shape[1]
            rho_pixel_values = np.logical_or(
                np.less_equal(rhos_pixel_values % 1, 10 ** (-10)),
                np.less_equal(1 - 10 ** (-10), rhos_pixel_values % 1),
            ).all()
            phi_pixel_values = np.logical_or(
                np.less_equal(phis_pixel_values % 1, 10 ** (-10)),
                np.less_equal(1 - 10 ** (-10), phis_pixel_values % 1),
            ).all()

            if not rho_pixel_values:
                raise ProbeSchemaError(
                    f"Choice for n_rows {self.params.n_rows} leads to interpolation errors, please change input variables"
                )
            if not phi_pixel_values:
                raise ProbeSchemaError(f"Choice for n_cols {self.params.n_cols} leads to interpolation errors")
        return rhos, phis

    @staticmethod
    def find_suitable_n_rows(
        row_min: int,
        row_max: int,
        length: int,
        boundary_condition: Union[
            Literal["periodic-symmetric", "periodic-left"],
            List[float],
        ] = "periodic_symmetric",
    ) -> List[int]:
        """Find proper spacing of rows/columns for given boundary conditions (i.e. image size, offset. etc).

        Args:
            row_min (int): Starting value for row count
            row_max (int): End value for row count
            length (int): Pixels in the respective dimension
            boundary_condition (Union[Literal["periodic-symmetric", "periodic-left"], List[float]], optional):  Boundary conditions for the probing can either be periodic or non-periodic, if they are periodic, the distance from one row to the next must be the same also for the boundaries. Defaults to "periodic_symmetric".
            Else, no conditions for the boundaries are required. Options are:
                - 'periodic-symmetric': the first and the last row are placed with an offset to the
                            borders, that is half of the spacing of the two rows
                - 'periodic-left': the first row is at the border of the bottom of the image, while
                        the last row is one spacing apart from the top of the image
                - list with two values: in this case the an offset of value f1 and f2 is set on both ends, i.e. the
                        the sampling no longer goes from 0 to 1 ('no-offset') but instead from 0+f1 to 0-f2

        Returns:
            List[int]: List of all number of rows that does not lead to interpolation errors
        """
        suitable_values: List[int] = []
        # loop through all values and validate whether they are suitable
        for counter in range(row_min, row_max + 1):
            if boundary_condition == "periodic-symmetric":
                values = np.linspace(0, 1, counter, endpoint=False)
                values = values + (values[1] - values[0]) / 2
            if boundary_condition == "periodic-left":
                values = np.linspace(0, 1, counter, endpoint=False)
            if isinstance(boundary_condition, List):
                values = np.linspace(0 + boundary_condition[0], 1 - boundary_condition[1], counter, endpoint=True)

            pixel_values = values * length
            pixel_values_modulo = pixel_values % 1
            no_interpolation = np.less_equal(pixel_values_modulo, 10 ** (-10))
            no_interpolation = np.logical_or(no_interpolation, np.less_equal(1 - 10 ** (-10), pixel_values_modulo))
            no_interpolation = no_interpolation.all()

            if no_interpolation:
                suitable_values.append(counter)

        return suitable_values
