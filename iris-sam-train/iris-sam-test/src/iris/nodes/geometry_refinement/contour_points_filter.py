import cv2
import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import GeometryPolygons, NoiseMask
from iris.io.errors import GeometryRefinementError


class ContourPointNoiseEyeballDistanceFilter(Algorithm):
    """Implementation of point filtering algorithm that removes points which are to close to eyeball or noise.

    The role of this algorithm is to create a buffer around the pupil and iris polygons. This accounts for
    potential segmentation imprecisions, making the overall pipeline more robust against edge cases and out-of-distribution images.

    The buffer width is computed relatively to the iris diameter: `min_distance_to_noise_and_eyeball * iris_diameter`
    The trigger for this buffer are the eyeball boundary and the noise (e.g. eyelashes, specular reflection, etc.).
    """

    class Parameters(Algorithm.Parameters):
        """Default ContourPointToNoiseEyeballDistanceFilter parameters."""

        min_distance_to_noise_and_eyeball: float = Field(..., gt=0.0, lt=1.0)

    __parameters_type__ = Parameters

    def __init__(self, min_distance_to_noise_and_eyeball: float = 0.005) -> None:
        """Assign parameters.

        Args:
            min_distance_to_noise_and_eyeball (float, optional): Minimum distance to eyeball or noise expressed as a fraction of iris diameter length. Defaults to 0.025.
        """
        super().__init__(min_distance_to_noise_and_eyeball=min_distance_to_noise_and_eyeball)

    def run(self, polygons: GeometryPolygons, geometry_mask: NoiseMask) -> GeometryPolygons:
        """Perform polygon refinement by filtering out those iris/pupil polygons points which are to close to eyeball or noise.

        Args:
            polygons (GeometryPolygons): Polygons to refine.
            geometry_mask (NoiseMask): Geometry noise mask.

        Returns:
            GeometryPolygons: Refined geometry polygons.
        """
        noise_and_eyeball_polygon_points_mask = geometry_mask.mask.copy()

        for eyeball_point in np.round(polygons.eyeball_array).astype(int):
            x, y = eyeball_point
            noise_and_eyeball_polygon_points_mask[y, x] = True

        min_dist_to_noise_and_eyeball_in_px = round(
            self.params.min_distance_to_noise_and_eyeball * polygons.iris_diameter
        )

        forbidden_touch_map = cv2.blur(
            noise_and_eyeball_polygon_points_mask.astype(float),
            ksize=(
                2 * min_dist_to_noise_and_eyeball_in_px + 1,
                2 * min_dist_to_noise_and_eyeball_in_px + 1,
            ),
        )
        forbidden_touch_map = forbidden_touch_map.astype(bool)

        return GeometryPolygons(
            pupil_array=self._filter_polygon_points(forbidden_touch_map, polygons.pupil_array),
            iris_array=self._filter_polygon_points(forbidden_touch_map, polygons.iris_array),
            eyeball_array=polygons.eyeball_array,
        )

    def _filter_polygon_points(self, forbidden_touch_map: np.ndarray, polygon_points: np.ndarray) -> np.ndarray:
        """Filter polygon's points.

        Args:
            forbidden_touch_map (np.ndarray): Forbidden touch map. If value of an element is greater then 0 then it means that point is to close to noise or eyeball.
            polygon_points (np.ndarray): Polygon's points.

        Returns:
            np.ndarray: Filtered polygon's points.
        """
        valid_points = [not forbidden_touch_map[y, x] for x, y in np.round(polygon_points).astype(int)]
        if not any(valid_points):
            raise GeometryRefinementError("No valid points after filtering polygon points!")

        return polygon_points[valid_points]
