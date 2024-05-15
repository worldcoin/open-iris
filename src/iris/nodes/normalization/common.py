from typing import Tuple

import numpy as np
from pydantic import NonNegativeInt

from iris.io.dataclasses import GeometryPolygons
from iris.utils import common


def generate_iris_mask(extrapolated_contours: GeometryPolygons, noise_mask: np.ndarray) -> np.ndarray:
    """Generate iris mask by first finding the intersection region between extrapolated iris contours and eyeball contours. Then remove from the outputted mask those pixels for which noise_mask is equal to True.

    Args:
        extrapolated_contours (GeometryPolygons): Iris polygon vertices.
        noise_mask (np.ndarray): Noise mask.

    Returns:
        np.ndarray: Iris mask.
    """
    img_h, img_w = noise_mask.shape[:2]

    iris_mask = common.contour_to_mask(extrapolated_contours.iris_array, (img_w, img_h))
    eyeball_mask = common.contour_to_mask(extrapolated_contours.eyeball_array, (img_w, img_h))

    iris_mask = iris_mask & eyeball_mask
    iris_mask = ~(iris_mask & noise_mask) & iris_mask

    return iris_mask


def correct_orientation(
    pupil_points: np.ndarray, iris_points: np.ndarray, eye_orientation: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Correct orientation by changing the starting angle in pupil and iris points' arrays.

    Args:
        pupil_points (np.ndarray): Pupil boundary points' array. NumPy array of shape (num_points = 512, xy_coords = 2).
        iris_points (np.ndarray): Iris boundary points' array. NumPy array of shape (num_points = 512, xy_coords = 2).
        eye_orientation (float): Eye orientation angle in radians.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple with rotated based on eye_orientation angle boundary points (pupil_points, iris_points).
    """
    orientation_angle = np.degrees(eye_orientation)
    num_rotations = -round(orientation_angle * len(pupil_points) / 360.0)

    pupil_points = np.roll(pupil_points, num_rotations, axis=0)
    iris_points = np.roll(iris_points, num_rotations, axis=0)

    return pupil_points, iris_points


def getgrids(res_in_r: NonNegativeInt, p2i_ratio: NonNegativeInt) -> np.ndarray:
    """Generate radius grids for nonlinear normalization based on p2i_ratio (pupil_to_iris ratio).

    Args:
        res_in_r (NonNegativeInt): Normalized image r resolution.
        p2i_ratio (NonNegativeInt): pupil_to_iris ratio, range in [0,100]

    Returns:
        np.ndarray: nonlinear sampling grids for normalization
    """
    p = [np.square(x) for x in np.arange(28, max(74 - p2i_ratio, p2i_ratio - 14), 1)]
    q = p - p[0]
    q = q / q[-1]
    grids = np.interp(np.linspace(0, 1.0, res_in_r + 1), np.linspace(0, 1.0, len(q)), q)
    return grids[0:-1] + np.diff(grids) / 2


def interpolate_pixel_intensity(image: np.ndarray, pixel_coords: Tuple[float, float]) -> float:
    """Perform bilinear interpolation to estimate pixel intensity in a given location.

    Args:
        image (np.ndarray): Original, not normalized image.
        pixel_coords (Tuple[float, float]): Pixel coordinates.

    Returns:
        float: Interpolated pixel intensity.

    Reference:
        [1] https://en.wikipedia.org/wiki/Bilinear_interpolation
    """

    def get_pixel_intensity(image: np.ndarray, pixel_x: float, pixel_y: float) -> float:
        """Get the intensity value of a pixel from an intensity image.

        Args:
            image (np.ndarray): Intensity image.
            pixel_x (float): Pixel x coordinate.
            pixel_y (float): Pixel y coordinate.

        Returns:
            float: Pixel value.
        """
        try:
            return image[int(pixel_y), int(pixel_x)]
        except IndexError:
            return 0.0

    def get_interpolation_points_coords(
        image: np.ndarray, pixel_x: float, pixel_y: float
    ) -> Tuple[float, float, float, float]:
        """Extract interpolation points coordinates.

        Args:
            image (np.ndarray): Original, not normalized image.
            pixel_x (float): Pixel x coordinate.
            pixel_y (float): Pixel y coordinate.

        Returns:
            Tuple[float, float, float, float]: Tuple with interpolation points coordinates in a format (xmin, ymin, xmax, ymax).
        """
        xmin, ymin = np.floor(pixel_x), np.floor(pixel_y)
        xmax, ymax = np.ceil(pixel_x), np.ceil(pixel_y)

        img_h, img_w = image.shape[:2]
        if xmin == xmax and not xmax == img_w - 1:
            xmax += 1
        if xmin == xmax and xmax == img_w - 1:
            xmin -= 1

        if ymin == ymax and not ymax == img_h - 1:
            ymax += 1
        if ymin == ymax and ymax == img_h - 1:
            ymin -= 1

        return xmin, ymin, xmax, ymax

    pixel_x, pixel_y = pixel_coords
    xmin, ymin, xmax, ymax = get_interpolation_points_coords(image, pixel_x=pixel_x, pixel_y=pixel_y)

    lower_left_pixel_intensity = get_pixel_intensity(image, pixel_x=xmin, pixel_y=ymax)

    lower_right_pixel_intensity = get_pixel_intensity(image, pixel_x=xmax, pixel_y=ymax)

    upper_left_pixel_intensity = get_pixel_intensity(image, pixel_x=xmin, pixel_y=ymin)

    upper_right_pixel_intensity = get_pixel_intensity(image, pixel_x=xmax, pixel_y=ymin)

    xs_differences = np.array([xmax - pixel_x, pixel_x - xmin])
    neighboring_pixel_intensities = np.array(
        [
            [lower_left_pixel_intensity, upper_left_pixel_intensity],
            [lower_right_pixel_intensity, upper_right_pixel_intensity],
        ]
    )
    ys_differences = np.array([[pixel_y - ymin], [ymax - pixel_y]])

    pixel_intensity = np.matmul(np.matmul(xs_differences, neighboring_pixel_intensities), ys_differences)

    return pixel_intensity.item()


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Map normalized image values from [0, 1] range to [0, 255] and cast dtype to np.uint8.

    Args:
        image (np.ndarray): Normalized iris.

    Returns:
        np.ndarray: Normalized iris with modified values.
    """
    out_image = np.round(image * 255)
    out_image = out_image.astype(np.uint8)

    return out_image
