from numbers import Number
from typing import Tuple, Union

import numpy as np
from pydantic import validator

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import BoundingBox, GeometryPolygons, IRImage
from iris.io.errors import BoundingBoxEstimationError
from iris.io.validators import are_all_positive


class IrisBBoxCalculator(Algorithm):
    """Calculate the smallest bounding box around the iris polygon, cropped or not, padded or not."""

    class Parameters(Algorithm.Parameters):
        """Parameters of the iris bounding box calculator."""

        buffer: Union[int, float, Tuple[Number, Number]]
        crop: bool

        _are_all_positive = validator("buffer", allow_reuse=True)(are_all_positive)

    __parameters_type__ = Parameters

    def __init__(self, buffer: Union[int, float, Tuple[Number, Number]] = 0, crop: bool = False) -> None:
        """Assign parameters.

        WARNING: Depending on wether `buffer` is a float or an int, behaviour differs.
        `buffer=2.0` => the iris size will be multiplied by 2., `buffer=2` => 2 pixels padding will be added.

        Args:
            buffer (Union[int, float, Tuple[Number, Number]], optional): Iris buffer in pixels.
                if `int`, the bounding box will be padded by `buffer` pixels in each direction.
                if `float`, the bounding box' height and width will be multiplied by `buffer`.
                if `Tuple[int]`, the bounding box will be padded by `buffer[0]` pixels in the x direction
                (left and right) and `buffer[1]` pixels in the y direction (top and bottom).
                if `Tuple[float]`, the bounding box width will be multiplied by `buffer[0]` and height by `buffer[1]`.
            crop (bool, optional): If True, the bounding box will be cropped to the shape of the input IR Image. Defaults to False.
        """
        super().__init__(buffer=buffer, crop=crop)

    def run(self, ir_image: IRImage, geometry_polygons: GeometryPolygons) -> BoundingBox:
        """Compute the bounding box around the iris with an additional buffer. Works best on extrapolated polygons.

        The buffer's behaviour is explained in the constructor's docstring.
        The bounding box will be cropped to the shape of the input IR Image.

        Args:
            ir_image (IRImage): IR image.
            geometry_polygons (GeometryPolygons): polygons, from which the iris polygon (respectively the image shape) used to compute the bounding box (resp. crop the bounding box).

        Returns:
            BoundingBox: Estimated iris bounding box.
        """
        iris_polygon = geometry_polygons.iris_array
        image_height, image_width = (ir_image.height, ir_image.width)
        buffer = (
            (self.params.buffer, self.params.buffer)
            if isinstance(self.params.buffer, (int, float))
            else self.params.buffer
        )

        original_x_min: float = np.min(iris_polygon[:, 0])
        original_x_max: float = np.max(iris_polygon[:, 0])
        original_y_min: float = np.min(iris_polygon[:, 1])
        original_y_max: float = np.max(iris_polygon[:, 1])

        if original_x_max == original_x_min or original_y_max == original_y_min:
            raise BoundingBoxEstimationError(
                f"Iris bounding box empty. x_min={original_x_min}, x_max={original_x_max}, "
                f"y_min={original_y_min}, y_max={original_y_max}"
            )

        if isinstance(buffer[0], int):
            padded_x_min = original_x_min - buffer[0]
            padded_x_max = original_x_max + buffer[0]
        else:
            bbox_width = original_x_max - original_x_min
            padded_x_min = original_x_min - bbox_width * (buffer[0] - 1) / 2
            padded_x_max = original_x_max + bbox_width * (buffer[0] - 1) / 2

        if isinstance(buffer[1], int):
            padded_y_min = original_y_min - buffer[1]
            padded_y_max = original_y_max + buffer[1]
        else:
            bbox_height = original_y_max - original_y_min
            padded_y_min = original_y_min - bbox_height * (buffer[1] - 1) / 2
            padded_y_max = original_y_max + bbox_height * (buffer[1] - 1) / 2

        if self.params.crop:
            padded_x_min = max(padded_x_min, 0)
            padded_x_max = min(padded_x_max, image_width)
            padded_y_min = max(padded_y_min, 0)
            padded_y_max = min(padded_y_max, image_height)

        return BoundingBox(x_min=padded_x_min, x_max=padded_x_max, y_min=padded_y_min, y_max=padded_y_max)
