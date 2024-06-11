from typing import Collection, List, Tuple

import cv2
import numpy as np
from pydantic import Field, validator

from iris.io.class_configs import Algorithm
from iris.io.dataclasses import EyeOrientation, GeometryPolygons, IRImage, NoiseMask, NormalizedIris
from iris.io.errors import NormalizationError
from iris.nodes.normalization.common import (
    correct_orientation,
    generate_iris_mask,
    get_pixel_or_default,
    interpolate_pixel_intensity,
    to_uint8,
)


class PerspectiveNormalization(Algorithm):
    """Implementation of a normalization algorithm which uses perspective transformation to map image pixels.

    Algorithm steps:
        1) Create a grid of trapezoids around iris in original image based on following algorithm parameters: res_in_phi, res_in_r, intermediate_radiuses.
        2) Create a grid of corresponding to each trapezoid rectangles in normalized image.
        3) For each corresponding trapezoid, rectangle pair compute perspective matrix to estimate normalized image pixel location in an original image location.
        4) Map each normalized image pixel to original image pixel based on estimated perspective matrix and perform bilinear interpolation if necessary.
    """

    class Parameters(Algorithm.Parameters):
        """Parameters class for PerspectiveNormalization."""

        res_in_phi: int = Field(..., gt=0)
        res_in_r: int = Field(..., gt=0)
        skip_boundary_points: int = Field(..., gt=0)
        intermediate_radiuses: Collection[float]
        oversat_threshold: int = Field(..., gt=0)

        @validator("intermediate_radiuses")
        def check_intermediate_radiuses(cls: type, v: Collection[float]) -> Collection[float]:
            """Check intermediate_radiuses parameter.

            Args:
                cls (type): PerspectiveNormalization.Parameters class.
                v (Collection[float]): Variable value to check.

            Raises:
                NormalizationError: Raised if number of radiuses is invalid or min value is less then 0.0 or greater than 1.0.

            Returns:
                Collection[float]: intermediate_radiuses value passed for further processing.
            """
            if len(v) < 2:
                raise NormalizationError(f"Invalid number of intermediate_radiuses: {len(v)}.")
            if min(v) < 0.0:
                raise NormalizationError(f"Invalid min value of intermediate_radiuses: {min(v)}.")
            if max(v) > 1.0:
                raise NormalizationError(f"Invalid max value of intermediate_radiuses: {max(v)}.")

            return v

    __parameters_type__ = Parameters

    def __init__(
        self,
        res_in_phi: int = 1024,
        res_in_r: int = 128,
        skip_boundary_points: int = 10,
        intermediate_radiuses: Collection[float] = np.linspace(0.0, 1.0, 10),
        oversat_threshold: int = 254,
    ) -> None:
        """Assign parameters.

        Args:
            res_in_phi (int): Normalized image phi resolution. Defaults to 1024.
            res_in_r (int): Normalized image r resolution. Defaults to 128.
            skip_boundary_points (int, optional): Take every nth point from estimated boundaries when generating correspondences.
                Defaults to 10.
            intermediate_radiuses (t.Iterable[float], optional): Intermediate rings radiuses used to generate additional points for estimating transformations.
                Defaults to np.linspace(0.0, 1.0, 10).
            oversat_threshold (int, optional): threshold for masking over-satuated pixels. Defaults to 254.
        """
        super().__init__(
            res_in_phi=res_in_phi,
            res_in_r=res_in_r,
            skip_boundary_points=skip_boundary_points,
            intermediate_radiuses=intermediate_radiuses,
            oversat_threshold=oversat_threshold,
        )

    def run(
        self,
        image: IRImage,
        noise_mask: NoiseMask,
        extrapolated_contours: GeometryPolygons,
        eye_orientation: EyeOrientation,
    ) -> NormalizedIris:
        """Normalize iris using perspective transformation estimated for every region of an image separately.

        Args:
            image (IRImage): Input image to normalize.
            noise_mask (NoiseMask): Noise mask.
            extrapolated_contours (GeometryPolygons): Extrapolated contours.
            eye_orientation (EyeOrientation): Eye orientation angle.

        Returns:
            NormalizedIris: NormalizedIris object containing normalized image and iris mask.

        Raises:
            NormalizationError: Raised if amount of iris and pupil points is different.
        """
        if len(extrapolated_contours.pupil_array) != len(extrapolated_contours.iris_array):
            raise NormalizationError("Extrapolated amount of iris and pupil points must be the same.")

        pupil_points, iris_points = correct_orientation(
            extrapolated_contours.pupil_array,
            extrapolated_contours.iris_array,
            eye_orientation.angle,
        )

        iris_mask = generate_iris_mask(extrapolated_contours, noise_mask.mask)
        iris_mask[image.img_data >= self.params.oversat_threshold] = False

        src_points, dst_points = self._generate_correspondences(pupil_points, iris_points)

        normalized_iris = NormalizedIris(
            normalized_image=np.zeros((self.params.res_in_r, self.params.res_in_phi), dtype=np.uint8),
            normalized_mask=np.zeros((self.params.res_in_r, self.params.res_in_phi), dtype=bool),
        )
        self._run_core(image, iris_mask, src_points, dst_points, normalized_iris)

        return normalized_iris

    def _run_core(
        self,
        image: IRImage,
        iris_mask: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        normalized_iris: NormalizedIris,
    ) -> None:
        for angle_point_idx in range(src_points.shape[1] - 1):
            for ring_idx in range(src_points.shape[0] - 1):
                current_src, current_dst = self._correspondence_rois_coords(
                    angle_idx=angle_point_idx,
                    ring_idx=ring_idx,
                    src_points=src_points,
                    dst_points=dst_points,
                )
                xmin, ymin, xmax, ymax = self._bbox_coords(current_dst)

                normalized_image_roi, normalized_mask_roi = self._normalize_roi(
                    original_image=image.img_data,
                    iris_mask=iris_mask,
                    src_points=current_src.astype(np.float32),
                    dst_points=current_dst.astype(np.float32),
                    normalize_roi_output_shape=(ymax - ymin, xmax - xmin),
                )

                normalized_iris.normalized_image[ymin:ymax, xmin:xmax] = to_uint8(normalized_image_roi)
                normalized_iris.normalized_mask[ymin:ymax, xmin:xmax] = normalized_mask_roi

    def _generate_correspondences(
        self, pupil_points: np.ndarray, iris_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate correspondences between points in original image and normalized image.

        Args:
            pupil_points (np.ndarray): Pupil bounding points. NumPy array of shape (num_points = 360, xy_coords = 2).
            iris_points (np.ndarray): Iris bounding points. NumPy array of shape (num_points = 360, xy_coords = 2).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with generated correspondences.
        """
        pupil_points = pupil_points[:: self.params.skip_boundary_points]
        iris_points = iris_points[:: self.params.skip_boundary_points]

        src_points = []
        for radius in self.params.intermediate_radiuses:
            ring = pupil_points + radius * (iris_points - pupil_points)
            ring = np.vstack([ring, ring[0]])
            src_points.append(ring)
        src_points = np.array(src_points)

        num_rings, num_ring_points = src_points.shape[:2]
        dst_xs, dst_ys = np.meshgrid(
            np.linspace(0, self.params.res_in_phi, num_ring_points).astype(int),
            np.linspace(0, self.params.res_in_r, num_rings).astype(int),
        )
        dst_points = np.array([dst_xs, dst_ys]).transpose((1, 2, 0))

        return src_points, dst_points

    def _normalize_roi(
        self,
        original_image: np.ndarray,
        iris_mask: np.ndarray,
        src_points: np.ndarray,
        dst_points: np.ndarray,
        normalize_roi_output_shape: Tuple[float, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Normalize a single ROI of an image.

        Args:
            original_image (np.ndarray): Entire input image to normalize.
            iris_mask (np.ndarray): Iris class segmentation mask.
            src_points (np.ndarray): ROI's original input image points.
            dst_points (np.ndarray): ROI's normalized image points.
            normalize_roi_output_shape (t.Tuple[float, float]): Output shape of normalized ROI.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with normalized image and mask ROIs.
        """
        xmin, ymin, xmax, ymax = self._bbox_coords(dst_points)

        normalize_image_xs = np.arange(xmin, xmax)
        normalize_image_ys = np.arange(ymin, ymax)

        normalize_image_points = np.meshgrid(normalize_image_xs, normalize_image_ys)
        normalize_image_points = self.cartesian2homogeneous(normalize_image_points)

        perspective_mat = cv2.getPerspectiveTransform(dst_points, src_points)
        mapped_points = np.matmul(perspective_mat, normalize_image_points)
        mapped_points = self.homogeneous2cartesian(mapped_points)

        normalized_image_roi = np.zeros(normalize_roi_output_shape, dtype=np.float32)
        normalized_mask_roi = np.zeros(normalize_roi_output_shape, dtype=bool)

        for image_xy, normalized_xy in zip(mapped_points.T, normalize_image_points.T[..., :2]):
            norm_x, norm_y = normalized_xy.astype(int)

            shifted_y, shifted_x = norm_y - ymin, norm_x - xmin

            normalized_image_roi[shifted_y, shifted_x] = interpolate_pixel_intensity(
                original_image, pixel_coords=image_xy
            )

            normalized_mask_roi[shifted_y, shifted_x] = get_pixel_or_default(
                iris_mask, image_xy[0], image_xy[1], default=False
            )

        return normalized_image_roi / 255.0, normalized_mask_roi

    def _bbox_coords(self, norm_dst_points: np.ndarray) -> Tuple[int, int, int, int]:
        """Extract the bounding box of currently processed normalized image ROI.

        Args:
            norm_dst_points (np.ndarray): Normalized image ROI coordinates.

        Returns:
            Tuple[int, int, int, int]: Bounding box coordinates in form (xmin, ymin, xmax, ymax).
        """
        xmin, ymin = norm_dst_points[0].astype(int)
        xmax, ymax = norm_dst_points[-1].astype(int)

        return (xmin, ymin, xmax, ymax)

    def _correspondence_rois_coords(
        self,
        angle_idx: int,
        ring_idx: int,
        src_points: np.ndarray,
        dst_points: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a single correspondence ROIs between original image and normalized one based on angle index and ring index.

        Args:
            angle_idx (int): Boundary point angle index.
            ring_idx (int): Intermediate ring index.
            src_points (np.ndarray): All mapping points from an original image.
                NumPy array of shape (
                    num_intermediate_rings = self.intermediate_radiuses,
                    num_boundary_points = 360 // self.skip_boundary_points,
                    xy_coords = 2
                ).
            dst_points (np.ndarray): All mapping points from an normalized image.
                NumPy array of shape (
                    num_intermediate_rings = self.intermediate_radiuses,
                    num_boundary_points = 360 // self.skip_boundary_points,
                    xy_coords = 2
                ).

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with extracted from src_points and dst_points ROIs.
        """
        src_roi = src_points[ring_idx : ring_idx + 2, angle_idx : angle_idx + 2]
        dst_roi = dst_points[ring_idx : ring_idx + 2, angle_idx : angle_idx + 2]

        return src_roi.reshape(4, 2), dst_roi.reshape(4, 2)

    @staticmethod
    def cartesian2homogeneous(points: List[np.ndarray]) -> np.ndarray:
        """Convert points in cartesian coordinates to homogeneous coordinates.

        Args:
            points (List[np.ndarray]): Points in cartesian coordinates. Array should be in format: [[x values], [y values]].

        Returns:
            np.ndarray: Points in homogeneous coordinates. Returned array will have format: [[x values], [y values], [1 ... 1]].
        """
        x_coords, y_coords = points

        x_coords = x_coords.reshape(-1, 1)
        y_coords = y_coords.reshape(-1, 1)

        homogeneous_coords = np.hstack([x_coords, y_coords, np.ones((len(x_coords), 1))])

        return homogeneous_coords.T

    @staticmethod
    def homogeneous2cartesian(points: np.ndarray) -> np.ndarray:
        """Convert points in homogeneous coordinates to cartesian coordinates.

        Args:
            points (np.ndarray): Points in homogeneous coordinates. Array should be in format: [[x values], [y values], [perspective scale values]].

        Returns:
            np.ndarray: Points in cartesian coordinates. Returned array will have format: [[x values], [y values]].
        """
        points /= points[-1]
        points = points[:2]

        return points
