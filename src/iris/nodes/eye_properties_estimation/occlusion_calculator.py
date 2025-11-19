from typing import List

import numpy as np
from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, EyeOcclusion, EyeOrientation, GeometryPolygons, NoiseMask
from iris.utils import common


class OcclusionCalculator(Algorithm):
    """Calculate the eye occlusion value.

    This algorithm computes the fraction of visible iris in an image based on extrapolated polygons and the various noise masks.
    For an occlusion of 0, the iris is completely occluded. For an occlusion of 1, the iris is completely visible
    For historical reasons, this remained called "Occlusion", while it more precisely refers to the "Opening" of the eye.

    The parameter `quantile_angle` refers to the zone of the iris to consider for the occlusion computation.
    This is because the middle horizontal third of the iris is usually more useful, since less likely to be occluded by the eyelids.
    For a `quantile_angle` of 90º, the entire iris will be considered.
    For a `quantile_angle` of 30º, the horizontal middle third of the iris will be considered.
    For a `quantile_angle` of 0º, nothing will be considered (limit value).
    """

    class Parameters(Algorithm.Parameters):
        """Default OcclusionCalculator parameters."""

        quantile_angle: float = Field(..., ge=0.0, le=90.0)

    __parameters_type__ = Parameters

    def __init__(self, quantile_angle: float, callbacks: List[Callback] = []) -> None:
        """Assign parameters.

        Args:
            quantile_angle (float): Quantile angle for estimating the area in which we want to calculate the visible fraction value in degrees.
            callbacks (List[Callback]): callbacks list. Defaults to [].
        """
        super().__init__(quantile_angle=quantile_angle, callbacks=callbacks)

    def run(
        self,
        extrapolated_polygons: GeometryPolygons,
        noise_mask: NoiseMask,
        eye_orientation: EyeOrientation,
        eye_centers: EyeCenters,
    ) -> EyeOcclusion:
        """Compute the iris visible fraction.

        Args:
            extrapolated_polygons (GeometryPolygons): Extrapolated polygons contours.
            noise_mask (NoiseMask): Noise mask.
            eye_orientation (EyeOrientation): Eye orientation angle.

        Returns:
            EyeOcclusion: Visible iris fraction.
        """
        if self.params.quantile_angle == 0.0:
            return EyeOcclusion(visible_fraction=0.0)
        img_h, img_w = noise_mask.mask.shape
        frame = np.array([[0, 0], [0, img_h], [img_w, img_h], [img_w, 0], [0, 0]])

        # Offset all points by the minimum value to avoid negative indices
        all_points = np.concatenate(
            [
                extrapolated_polygons.iris_array,
                extrapolated_polygons.eyeball_array,
                extrapolated_polygons.pupil_array,
                frame,
            ]
        )
        offset = np.floor(all_points.min(axis=0))  # Negative or null
        total_mask_shape = (np.ceil(all_points.max(axis=0)) - offset).astype(int)

        overflow = np.ceil(all_points.max(axis=0)) - (img_w, img_h)
        pads = np.array([(-offset[1], overflow[1]), (-offset[0], overflow[0])]).astype(int)
        offseted_noise_mask = np.pad(noise_mask.mask, pads)

        iris_quantile_polygon = self._get_quantile_points(extrapolated_polygons.iris_array, eye_orientation)
        iris_mask_quantile = common.contour_to_mask(iris_quantile_polygon - offset, mask_shape=total_mask_shape)
        pupil_mask = common.contour_to_mask(extrapolated_polygons.pupil_array - offset, mask_shape=total_mask_shape)
        eyeball_mask = common.contour_to_mask(extrapolated_polygons.eyeball_array - offset, mask_shape=total_mask_shape)
        frame_mask = common.contour_to_mask(frame - offset, mask_shape=total_mask_shape)

        visible_iris_mask = iris_mask_quantile & ~pupil_mask & eyeball_mask & ~offseted_noise_mask & frame_mask
        extrapolated_iris_mask = iris_mask_quantile & ~pupil_mask

        if extrapolated_iris_mask.sum() == 0:
            return EyeOcclusion(visible_fraction=0.0)

        visible_fraction = visible_iris_mask.sum() / extrapolated_iris_mask.sum()

        return EyeOcclusion(visible_fraction=visible_fraction)

    def _get_quantile_points(
        self,
        iris_coords: np.ndarray,
        eye_orientation: EyeOrientation,
    ) -> np.ndarray:
        """Get those iris's points which fall into a specified quantile.

        Args:
            iris_coords (np.ndarray): Iris polygon coordinates.
            eye_orientation: (EyeOrientation): Eye orientation.

        Returns:
            np.ndarray: Iris polygon coordinates that fall within the quantile regions.
        """
        orientation_angle = np.degrees(eye_orientation.angle)
        num_rotations = -round(orientation_angle * len(iris_coords) / 360.0)
        iris_coords = np.roll(iris_coords, num_rotations, axis=0)

        scaled_quantile = round(self.params.quantile_angle * len(iris_coords) / 360.0)
        iris_quantile_coords = np.concatenate(
            [
                iris_coords[0:scaled_quantile],
                iris_coords[len(iris_coords) // 2 - scaled_quantile : len(iris_coords) // 2 + scaled_quantile],
                iris_coords[len(iris_coords) - scaled_quantile :],
            ]
        )

        return iris_quantile_coords
