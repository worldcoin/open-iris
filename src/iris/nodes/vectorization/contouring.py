from typing import Callable, List

import cv2
import numpy as np
from pydantic import NonNegativeFloat

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import GeometryMask, GeometryPolygons
from iris.io.errors import VectorizationError
from iris.utils.math import area


def filter_polygon_areas(
    polygons: List[np.ndarray], rel_tr: NonNegativeFloat = 0.03, abs_tr: NonNegativeFloat = 0.0
) -> List[np.ndarray]:
    """Filter out polygons whose area is below either an absolute threshold or a fraction of the largest area.

    Args:
        polygons (List[np.ndarray]): List of polygons to filter.
        rel_tr (NonNegativeFloat, optional): Relative threshold. Defaults to 0.03.
        abs_tr (NonNegativeFloat, optional): Absolute threshold. Defaults to 0.0.

    Returns:
        List[np.ndarray]: Filtered polygons' list.
    """
    areas = [area(polygon) if len(polygon) > 2 else 1.0 for polygon in polygons]
    area_factors = np.array(areas) / np.max(areas)

    filtered_polygons = [
        polygon
        for area, area_factor, polygon in zip(areas, area_factors, polygons)
        if area > abs_tr and area_factor > rel_tr
    ]

    return filtered_polygons


class ContouringAlgorithm(Algorithm):
    """Implementation of a vectorization process through contouring raster image."""

    class Parameters(Algorithm.Parameters):
        """Parameters class of the ContouringAlgorithm class."""

        contour_filters: List[Callable[[List[np.ndarray]], List[np.ndarray]]]

    __parameters_type__ = Parameters

    def __init__(
        self,
        contour_filters: List[Callable[[List[np.ndarray]], List[np.ndarray]]] = [filter_polygon_areas],
    ) -> None:
        """Assign parameters.

        Args:
            contour_filters (List[Callable[[List[np.ndarray]], List[np.ndarray]]], optional): List of filter functions used to filter out noise in polygons.
                Defaults to [ContouringAlgorithm.filter_polygon_areas].
        """
        super().__init__(contour_filters=contour_filters)

    def run(self, geometry_mask: GeometryMask) -> GeometryPolygons:
        """Contouring vectorization algorithm implementation.

        Args:
            geometry_mask (GeometryMask): Geometry segmentation map.

        Raises:
            VectorizationError: Raised if iris region not segmented or an error occur during iris region processing.

        Returns:
            GeometryPolygons: Geometry polygons points.
        """
        if not np.any(geometry_mask.iris_mask):
            raise VectorizationError("Geometry raster verification failed.")

        geometry_contours = self._find_contours(geometry_mask)

        return geometry_contours

    def _find_contours(self, mask: GeometryMask) -> GeometryPolygons:
        """Find raw contours for different classes in raster.

        Args:
            mask (GeometryMask): Raster object.

        Returns:
            GeometryPolygons: Raw contours indicating polygons of different classes.
        """
        eyeball_array = self._find_class_contours(mask.filled_eyeball_mask.astype(np.uint8))
        iris_array = self._find_class_contours(mask.filled_iris_mask.astype(np.uint8))
        pupil_array = self._find_class_contours(mask.pupil_mask.astype(np.uint8))

        return GeometryPolygons(pupil_array=pupil_array, iris_array=iris_array, eyeball_array=eyeball_array)

    def _find_class_contours(self, binary_mask: np.ndarray) -> np.ndarray:
        """Find contour between two different contours.

        Args:
            binary_mask (np.ndarray): Raster object.

        Raises:
            VectorizationError: Raised if number of contours found is different than 1.

        Returns:
            np.ndarray: Contour points array.
        """
        contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if hierarchy is None:
            raise VectorizationError("_find_class_contours: No contour hierarchy found at all.")

        parent_indices = np.flatnonzero(hierarchy[..., 3] == -1)
        contours = [np.squeeze(contours[i]) for i in parent_indices]

        contours = self._filter_contours(contours)

        if len(contours) != 1:
            raise VectorizationError("_find_class_contours: Number of contours must be equal to 1.")

        return contours[0]

    def _filter_contours(self, contours: List[np.ndarray]) -> List[np.ndarray]:
        """Filter contours based on predefined filters.

        Args:
            contours (List[np.ndarray]): Contours list.

        Returns:
            List[np.ndarray]: Filtered list of contours.
        """
        for filter_func in self.params.contour_filters:
            contours = filter_func(contours)

        return contours
