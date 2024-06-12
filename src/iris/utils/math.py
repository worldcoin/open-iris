import math
from typing import Dict, Tuple

import numpy as np


def area(array: np.ndarray, signed: bool = False) -> float:
    """Shoelace formula for simple polygon area calculation.

    WARNING: This formula only works for "simple polygons", i.e planar polygon without self-intersection nor holes.
    These conditions are not checked within this function.

    Args:
        array (np.ndarray): np array representing a polygon as a list of points, i.e. of shape (_, 2).
        signed (bool): If True, the area is signed, i.e. negative if the polygon is oriented clockwise.

    Returns:
        float: Polygon area

    Raises:
        ValueError: if the input array does not have shape (_, 2)

    References:
        [1] https://en.wikipedia.org/wiki/Shoelace_formula
        [2] https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    """
    if len(array.shape) != 2 or array.shape[1] != 2:
        raise ValueError(f"Unable to determine the area of a polygon with shape {array.shape}. Expecting (_, 2).")

    xs, ys = array.T
    area = 0.5 * (np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1)))
    if not signed:
        area = abs(area)

    return float(area)


def estimate_diameter(polygon: np.ndarray) -> float:
    """Estimates the diameter of an arbitrary arc by evaluating the maximum distance between any two points on the arc.

    Args:
        polygon (np.ndarray): Polygon points.

    Returns:
        float: Estimated diameter length.

    Reference:
        [1] https://sparrow.dev/pairwise-distance-in-numpy/
    """
    return float(np.linalg.norm(polygon[:, None, :] - polygon[None, :, :], axis=-1).max())


def cartesian2polar(xs: np.ndarray, ys: np.ndarray, center_x: float, center_y: float) -> Tuple[np.ndarray, np.ndarray]:
    """Convert xs and ys cartesian coordinates to polar coordinates.

    Args:
        xs (np.ndarray): x values.
        ys (np.ndarray): y values.
        center_x (float): center's x.
        center_y (float): center's y.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Converted coordinates (rhos, phis).
    """
    x_rel: np.ndarray = xs - center_x
    y_rel: np.ndarray = ys - center_y

    C = np.vectorize(complex)(x_rel, y_rel)

    rho = np.abs(C)
    phi = np.angle(C) % (2 * np.pi)

    return rho, phi


def polar2cartesian(
    rhos: np.ndarray, phis: np.ndarray, center_x: float, center_y: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert polar coordinates to cartesian coordinates.

    Args:
        rho (np.ndarray): rho values.
        phi (np.ndarray): phi values.
        center_x (float): center's x.
        center_y (float): center's y.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Converted coordinates (xs, ys).
    """
    xs = center_x + rhos * np.cos(phis)
    ys = center_y + rhos * np.sin(phis)

    return xs, ys


def orientation(moments: Dict[str, float]) -> float:
    """Compute the main orientation of a contour or a binary image given its precomputed cv2 moments.

    Args:
        moments (Dict[str, float]): cv2.moments of desired the binary image or contour.

    Returns:
        float: Main orientation of the shape. The orientation is a float in [-pi/2, pi/2[ representing the signed angle from the x axis.
    """
    # Edge case of null denominator
    if (moments["mu20"] - moments["mu02"]) == 0:
        if moments["mu11"] == 0:
            orientation = 0.0
        else:
            orientation = math.copysign(np.pi / 4, moments["mu11"])
    else:
        # General formula
        orientation = 0.5 * np.arctan(2 * moments["mu11"] / (moments["mu20"] - moments["mu02"]))
        if (moments["mu20"] - moments["mu02"]) < 0:
            orientation += np.pi / 2

        # Restricting the angle to [-pi/2, pi/2[
        orientation = np.mod(orientation + np.pi / 2, np.pi) - np.pi / 2

    return orientation


def eccentricity(moments: Dict[str, float]) -> float:
    r"""Compute the eccentricity of a contour or a binary image given its precomputed cv2 moments.

    The eccentricity is a number in [0, 1] which caracterises the "roundness" or "linearity" of a shape.
    A perfect circle will have an eccentricity of 0, and an infinite line an eccentricity of 1.
    For ellipses, the eccentricity is calculated as :math:`\frac{\sqrt{a^2 - b^2}}{a^2}`
    with a (resp. b) the semi-major (resp. -minor) axis of the ellipses.

    For `mu20 + mu02 == 0`, i.e. perfect line, the max theoretical value (1.0) is returned

    Args:
        moments (Dict[str, float]): cv2.moments of desired the binary image or contour.

    Returns:
        eccentricity (float): the eccentricity of the contour or binary map.

    Reference:
        [1] https://t1.daumcdn.net/cfile/tistory/15425F4150F4EBFC19
    """
    if moments["mu20"] + moments["mu02"] == 0:
        return 1.0

    # fmt: off
    eccentricity = ((moments["mu20"] - moments["mu02"]) ** 2 + 4 * moments["mu11"] ** 2) / (moments["mu20"] + moments["mu02"]) ** 2
    # fmt: on

    return eccentricity


def apply_weights_1d(scores_1d: np.ndarray, weights_1d: np.ndarray) -> float:
    """Apply weights for score fusion.

    Args:
        scores_1d (np.ndarray): scores to be fused.
        weights_1d (np.ndarray): weights.

    Raises:
        ValueError: if the input 1d arrays do not have the same length.

    Returns:
        float: fused score.
    """
    if len(scores_1d) != len(weights_1d):
        raise ValueError("Unable to apply weights. Dimension is different between scores and weights.")

    if len(weights_1d) == 0:
        raise ValueError("Unable to apply weights. Empty arrays.")

    if np.sum(weights_1d) == 0:
        raise ValueError("Unable to apply weights. Sum of weights is zero.")

    weighted_score = np.sum(np.multiply(scores_1d, weights_1d))

    return weighted_score / np.sum(weights_1d)


def polygon_length(polygon: np.ndarray, max_point_distance: int = 20) -> float:
    """Compute the length of a polygon represented as a (_, 2)-dimensionnal numpy array.

    One polygon can include several disjoint arcs, which should be identified as separate so that the distance
    between them is not counted. If a polygon is made of two small arc separated by a large distance, then the large
    distance between the two arcs will not be discounted in the polygon's length

    WARNING: The input polygon is assumed to be non-looped, i.e. if the first and last point are not equal,
    which is the case for all ou GeometryPolygons. The last implicit segment looping back from the
    last to the first point is therefore not included in the computed polygon length.

    Args:
        polygon (np.ndarray): (_, 2) - shaped numpy array representing a polygon.
        max_point_distance (int): Maximum distance between two points for them to be considered part of the same arc.

    Returns:
        float: length of the polygon, in pixels.
    """
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(f"This function expects a polygon, i.e. an array of shape (_, 2). Got {polygon.shape}")

    inter_point_distances = np.linalg.norm(np.roll(polygon, 1, axis=0) - polygon, axis=1)
    inter_point_distances = inter_point_distances[inter_point_distances < max_point_distance]

    return inter_point_distances.sum()
