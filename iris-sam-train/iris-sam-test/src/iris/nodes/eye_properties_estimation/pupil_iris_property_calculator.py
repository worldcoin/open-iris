from typing import List

from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons, PupilToIrisProperty
from iris.io.errors import PupilIrisPropertyEstimationError


class PupilIrisPropertyCalculator(Algorithm):
    """Computes pupil-to-iris properties.

    Algorithm steps:
        (1) Calculate pupil diameter to iris diameter ratio, i.e. pupil dilation.
        (2) Calculate the ratio of the pupil center to iris center distance over the iris diameter.
    """

    class Parameters(Algorithm.Parameters):
        """PupilIrisPropertyCalculator parameters.

        min_pupil_diameter (float): threshold of pupil diameter, below which the pupil is too small. min_pupil_diameter should be higher than 0.
        min_iris_diameter (float): threshold of iris diameter, below which the iris is too small. min_iris_diameter should be higher than 0.
        """

        min_pupil_diameter: float = Field(..., gt=0.0)
        min_iris_diameter: float = Field(..., gt=0.0)

    __parameters_type__ = Parameters

    def __init__(
        self,
        min_pupil_diameter: float = 1.0,
        min_iris_diameter: float = 150.0,
        callbacks: List[Callback] = [],
    ) -> None:
        """Assign parameters.

        Args:
            min_pupil_diameter (float): minimum pupil diameter. Defaults to 1.0.
            min_iris_diameter (float): minimum iris diameter. Defaults to 150.0.
            callbacks (List[Callback]): callbacks list. Defaults to [].
        """
        super().__init__(
            min_pupil_diameter=min_pupil_diameter,
            min_iris_diameter=min_iris_diameter,
            callbacks=callbacks,
        )

    def run(self, geometries: GeometryPolygons, eye_centers: EyeCenters) -> PupilToIrisProperty:
        """Calculate pupil-to-iris property.

        Args:
            geometries (GeometryPolygons): polygons used for calculating pupil-ro-iris property.
            eye_centers (EyeCenters): eye centers used for calculating pupil-ro-iris property.

        Raises:
            PupilIrisPropertyEstimationError: Raised if 1) the pupil or iris diameter is too small, 2) pupil diameter is larger than or equal to iris diameter, 3) pupil center is outside iris.

        Returns:
            PupilToIrisProperty: pupil-ro-iris property object.
        """
        iris_diameter = geometries.iris_diameter
        pupil_diameter = geometries.pupil_diameter

        if pupil_diameter < self.params.min_pupil_diameter:
            raise PupilIrisPropertyEstimationError("Pupil diameter is too small!")
        if iris_diameter < self.params.min_iris_diameter:
            raise PupilIrisPropertyEstimationError("Iris diameter is too small!")
        if pupil_diameter >= iris_diameter:
            raise PupilIrisPropertyEstimationError("Pupil diameter is larger than/equal to Iris diameter!")
        if eye_centers.center_distance * 2 >= iris_diameter:
            raise PupilIrisPropertyEstimationError("Pupil center is outside iris!")

        return PupilToIrisProperty(
            pupil_to_iris_diameter_ratio=pupil_diameter / iris_diameter,
            pupil_to_iris_center_dist_ratio=eye_centers.center_distance * 2 / iris_diameter,
        )
