from typing import List, Tuple

import numpy as np
from pydantic import Field

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeCenters, GeometryPolygons
from iris.io.errors import GeometryRefinementError
from iris.utils import math


class Smoothing(Algorithm):
    """Implementation of contour smoothing algorithm.

    Algorithm steps:
        1) Map iris/pupil points to polar space based on estimated iris/pupil centers.
        2) Smooth iris/pupil contour by applying 1D convolution with rolling median kernel approach.
        3) Map points back to cartesian space from polar space.
    """

    class Parameters(Algorithm.Parameters):
        """Smoothing parameters class."""

        dphi: float = Field(..., gt=0.0, lt=360.0)
        kernel_size: float = Field(..., gt=0.0, lt=360.0)
        gap_threshold: float = Field(..., gt=0.0, lt=360.0)

    __parameters_type__ = Parameters

    def __init__(self, dphi: float = 1.0, kernel_size: float = 10.0, gap_threshold: float = 10.0) -> None:
        """Assign parameters.

        Args:
            dphi (float, optional): phi angle delta used to sample points while doing smoothing by interpolation. Defaults to 1.0.
            kernel_size (float, optional): Rolling median kernel size expressed in radians. Final kernel size is computed as a quotient of kernel_size and dphi. Defaults to 10.0.
            gap_threshold (float, optional): Gap threshold distance. Defaults to None. Defaults to 10.0.
        """
        super().__init__(dphi=dphi, kernel_size=kernel_size, gap_threshold=gap_threshold)

    @property
    def kernel_offset(self) -> int:
        """Kernel offset (distance from kernel center to border) property used when smoothing with rolling median. If a quotient is less then 1 then kernel size equal to 1 is returned.

        Returns:
            int: Kernel size.
        """
        return max(1, int((np.radians(self.params.kernel_size) / np.radians(self.params.dphi))) // 2)

    def run(self, polygons: GeometryPolygons, eye_centers: EyeCenters) -> GeometryPolygons:
        """Perform smoothing refinement.

        Args:
            polygons (GeometryPolygons): Contours to refine.
            eye_centers (EyeCenters): Eye center used when performing a coordinates mapping from cartesian space to polar space.

        Returns:
            GeometryPolygons: Smoothed contours.
        """
        pupil_arcs = self._smooth(polygons.pupil_array, (eye_centers.pupil_x, eye_centers.pupil_y))
        iris_arcs = self._smooth(polygons.iris_array, (eye_centers.iris_x, eye_centers.iris_y))

        return GeometryPolygons(pupil_array=pupil_arcs, iris_array=iris_arcs, eyeball_array=polygons.eyeball_array)

    def _smooth(self, polygon: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
        """Smooth a single contour.

        Args:
            polygon (np.ndarray): Contour to smooth.
            center_xy (Tuple[float, float]): Contour's center.

        Returns:
            np.ndarray: Smoothed contour's vertices.
        """
        arcs, num_gaps = self._cut_into_arcs(polygon, center_xy)

        arcs = (
            self._smooth_circular_shape(arcs[0], center_xy)
            if num_gaps == 0
            else np.vstack([self._smooth_arc(arc, center_xy) for arc in arcs if len(arc) >= 2])
        )

        return arcs

    def _cut_into_arcs(self, polygon: np.ndarray, center_xy: Tuple[float, float]) -> Tuple[List[np.ndarray], int]:
        """Cut contour into arcs.

        Args:
            polygon (np.ndarray): Contour polygon.
            center_xy (Tuple[float, float]): Polygon's center.

        Returns:
            Tuple[List[np.ndarray], int]: Tuple with: (list of list of vertices, number of gaps detected in a contour).
        """
        rho, phi = math.cartesian2polar(polygon[:, 0], polygon[:, 1], *center_xy)
        phi, rho = self._sort_two_arrays(phi, rho)

        differences = np.abs(phi - np.roll(phi, -1))
        # True distance between first and last point
        differences[-1] = 2 * np.pi - differences[-1]

        gap_indices = np.argwhere(differences > np.radians(self.params.gap_threshold)).flatten()

        if gap_indices.size < 2:
            return [polygon], gap_indices.size

        gap_indices += 1
        phi, rho = np.split(phi, gap_indices), np.split(rho, gap_indices)

        arcs = [
            np.column_stack(math.polar2cartesian(rho_coords, phi_coords, *center_xy))
            for rho_coords, phi_coords in zip(rho, phi)
        ]

        # Connect arc which lies between 0 and 2π.
        if len(arcs) == gap_indices.size + 1:
            arcs[0] = np.vstack([arcs[0], arcs[-1]])
            arcs = arcs[:-1]

        return arcs, gap_indices.size

    def _smooth_arc(self, vertices: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
        """Smooth a single contour arc.

        Args:
            vertices (np.ndarray): Arc's vertices.
            center_xy (Tuple[float, float]): Center of an entire contour.

        Returns:
            np.ndarray: Smoothed arc's vertices.
        """
        rho, phi = math.cartesian2polar(vertices[:, 0], vertices[:, 1], *center_xy)
        phi, rho = self._sort_two_arrays(phi, rho)

        idx = self._find_start_index(phi)
        offset = phi[idx]
        relative_phi = (phi - offset) % (2 * np.pi)

        smoothed_relative_phi, smoothed_rho = self._smooth_array(relative_phi, rho)

        smoothed_phi = (smoothed_relative_phi + offset) % (2 * np.pi)

        x_smoothed, y_smoothed = math.polar2cartesian(smoothed_rho, smoothed_phi, *center_xy)

        return np.column_stack([x_smoothed, y_smoothed])

    def _smooth_circular_shape(self, vertices: np.ndarray, center_xy: Tuple[float, float]) -> np.ndarray:
        """Smooth arc in a form of a circular shape.

        Args:
            vertices (np.ndarray): Arc's vertices.
            center_xy (Tuple[float, float]): Center of an entire contour.

        Returns:
            np.ndarray: Smoothed arc's vertices.
        """
        rho, phi = math.cartesian2polar(vertices[:, 0], vertices[:, 1], *center_xy)

        padded_phi = np.concatenate([phi - 2 * np.pi, phi, phi + 2 * np.pi])
        padded_rho = np.concatenate([rho, rho, rho])

        smoothed_phi, smoothed_rho = self._smooth_array(padded_phi, padded_rho)

        mask = (smoothed_phi >= 0) & (smoothed_phi < 2 * np.pi)
        rho_smoothed, phi_smoothed = smoothed_rho[mask], smoothed_phi[mask]

        x_smoothed, y_smoothed = math.polar2cartesian(rho_smoothed, phi_smoothed, *center_xy)

        return np.column_stack([x_smoothed, y_smoothed])

    def _smooth_array(self, phis: np.ndarray, rhos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Smooth coordinates expressed in polar space.

        Args:
            phis (np.ndarray): phi values.
            rhos (np.ndarray): rho values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with smoothed coordinates (phis, rhos).
        """
        interpolated_phi = np.arange(min(phis), max(phis), np.radians(self.params.dphi))
        interpolated_rho = np.interp(interpolated_phi, xp=phis, fp=rhos, period=2 * np.pi)

        smoothed_rho = self._rolling_median(interpolated_rho, self.kernel_offset)
        if len(interpolated_phi) - 1 >= self.kernel_offset * 2:
            smoothed_phi = interpolated_phi[self.kernel_offset : -self.kernel_offset]
        else:
            smoothed_phi = interpolated_phi

        return smoothed_phi, smoothed_rho

    def _sort_two_arrays(self, first_list: np.ndarray, second_list: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Sort both numpy arrays based on values from the first_list.

        Args:
            first_list (np.ndarray): First array.
            second_list (np.ndarray): Second array.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with (sorted first array, sorted second array).
        """
        zipped_lists = zip(first_list, second_list)
        sorted_pairs = sorted(zipped_lists)

        sorted_tuples = zip(*sorted_pairs)
        first_list, second_list = [list(sorted_tuple) for sorted_tuple in sorted_tuples]

        return np.array(first_list), np.array(second_list)

    def _find_start_index(self, phi: np.ndarray) -> int:
        """Find the start index by checking the largest gap. phi needs to be sorted.

        Args:
            phi (np.ndarray): phi angle values.

        Raises:
            GeometryRefinementError: Raised if phi values are not sorted ascendingly.

        Returns:
            int: Index value.
        """
        if not np.all((phi - np.roll(phi, 1))[1:] >= 0):
            raise GeometryRefinementError("Smoothing._find_start_index phi must be sorted ascendingly!")

        phi_tmp = np.concatenate(([phi[-1] - 2 * np.pi], phi, [phi[0] + 2 * np.pi]))
        phi_tmp_left_neighbor = np.roll(phi_tmp, 1)
        dphi = (phi_tmp - phi_tmp_left_neighbor)[1:-1]
        largest_gap_index = np.argmax(dphi)

        return int(largest_gap_index)

    def _rolling_median(self, signal: np.ndarray, kernel_offset: int) -> np.ndarray:
        """Compute rolling median of a 1D signal.

        Args:
            signal (np.ndarray): Signal values.
            kernel_size (int): Kernel size.

        Raises:
            GeometryRefinementError: Raised if signal is not 1D.

        Returns:
            np.ndarray: Rolling median result.
        """
        if signal.ndim != 1:
            raise GeometryRefinementError("Smoothing._rolling_median only works for 1d arrays.")

        stacked_signals: List[np.ndarray] = []
        for i in range(-kernel_offset, kernel_offset + 1):
            stacked_signals.append(np.roll(signal, i))
        stacked_signals = np.stack(stacked_signals)

        rolling_median = np.median(stacked_signals, axis=0)
        if len(rolling_median) - 1 >= kernel_offset * 2:
            rolling_median = rolling_median[kernel_offset:-kernel_offset]

        return rolling_median
