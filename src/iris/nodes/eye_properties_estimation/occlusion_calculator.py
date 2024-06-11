from typing import List, Tuple

import numpy as np
from pydantic import Field

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, EyeOcclusion, EyeOrientation, GeometryPolygons, NoiseMask
from iris.utils import common, math


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
            eye_centers (EyeCenters): Eye centers.

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

        xs2mask, ys2mask = self._get_quantile_points(extrapolated_polygons.iris_array, eye_orientation, eye_centers)
        iris_mask_quantile = common.contour_to_mask(
            np.column_stack([xs2mask, ys2mask]) - offset, mask_shape=total_mask_shape
        )
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
        self, iris_coords: np.ndarray, eye_orientation: EyeOrientation, eye_centers: EyeCenters
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get those iris's points which fall into a specified quantile.

        Args:
            iris_coords (np.ndarray): Iris polygon coordinates.
            eye_orientation: (EyeOrientation): Eye orientation.
            eye_centers: (EyeCenters): Eye centers.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with xs and ys that falls into quantile region.
        """
        orientation_angle = np.degrees(eye_orientation.angle)
        num_rotations = -round(orientation_angle * len(iris_coords) / 360.0)

        iris_xs, iris_ys = iris_coords[:, 0], iris_coords[:, 1]
        iris_rhos, iris_phis = math.cartesian2polar(iris_xs, iris_ys, eye_centers.iris_x, eye_centers.iris_y)

        iris_phis = np.roll(iris_phis, num_rotations, axis=0)
        iris_rhos = np.roll(iris_rhos, num_rotations, axis=0)

        scaled_quantile = round(self.params.quantile_angle * len(iris_coords) / 360.0)

        phis2mask = np.concatenate(
            [
                iris_phis[:scaled_quantile],
                iris_phis[-scaled_quantile:],
                iris_phis[len(iris_phis) // 2 : len(iris_phis) // 2 + scaled_quantile],
                iris_phis[len(iris_phis) // 2 - scaled_quantile : len(iris_phis) // 2],
            ]
        )
        rhos2mask = np.concatenate(
            [
                iris_rhos[:scaled_quantile],
                iris_rhos[-scaled_quantile:],
                iris_rhos[len(iris_rhos) // 2 : len(iris_rhos) // 2 + scaled_quantile],
                iris_rhos[len(iris_rhos) // 2 - scaled_quantile : len(iris_rhos) // 2],
            ]
        )
        phis2mask, rhos2mask = zip(*sorted(zip(phis2mask, rhos2mask)))
        xs2mask, ys2mask = math.polar2cartesian(rhos2mask, phis2mask, eye_centers.iris_x, eye_centers.iris_y)

        return xs2mask, ys2mask
