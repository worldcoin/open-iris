from typing import Any, Dict, List

import cv2
import numpy as np


def generate_arc(
    radius: float, center_x: float, center_y: float, from_angle: float, to_angle: float, num_points: int = 1000
) -> np.ndarray:
    angles = np.linspace(from_angle, to_angle, num_points, endpoint=not (from_angle == 0.0 and to_angle == 2 * np.pi))

    circle_xs = radius * np.cos(angles) + center_x
    circle_ys = radius * np.sin(angles) + center_y

    return np.column_stack([circle_xs, circle_ys])


def generate_multiple_arcs(arc_params: List[Dict[str, Any]]) -> np.ndarray:
    return np.concatenate([generate_arc(**kwargs) for kwargs in arc_params])


def rotated_elliptical_contour(
    theta: float, a: float = 5, b: float = 1, resolution: int = 200, centered=False
) -> np.ndarray:
    r"""Compute the pixellised contour of a rotated ellipses.

    This function creates a binary image where   :math:`pixel = 1 \Leftrightarrow pixel \in` ellipse

    Ellipse equation

    :math:`(\frac{x}{a})^2 + (\frac{y}{b})^2 < 1`

    Rotate by :math:`\theta`

    :math:`(\frac{x cos(\theta) + y sin(\theta)}{a})^2 + (\frac{x sin(\theta) - y cos(\theta)}{b})^2 < 1`

    Isolate x and y

    :math:`((\frac{cos(\theta)}{b})^2 + (\frac{sin(\theta)}{a})^2)x^2 + 2 cos(\theta)sin(\theta)(b^2 - a^2)xy
    +((\frac{sin(\theta)}{b})^2 + (\frac{cos(\theta)}{a})^2)y^2 < a^2b^2 \blacksquare`

    Source: :math:`math`

    Or https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html because if it's on internet it's true.

    Also, `resolution` determines the precision of the contour by being the side of the square binary image used to
    generate contour, but also the diameter of the final ellipsis

    Args:
        theta (float): angle between the x axis and the major-axis of the ellipses
        a (float): The semi-major axis of the ellipses. Must be below 10, or the ellipse could crop out of the image.
        b (float): The semi-minor axis of the ellipses. Must be below 10, or the ellipse could crop out of the image.
        resolution (int): side of the square binary image used to generate contour

    Returns:
        np.ndarray: produced contour of shape (_, 1, 2)
    """
    x, y = np.meshgrid(np.linspace(-10, 10, resolution), np.linspace(-10, 10, resolution))
    x, y = x.flatten(), y.flatten()

    binary_map = (
        ((a * np.sin(theta)) ** 2 + (b * np.cos(theta)) ** 2) * x**2
        + (2 * (b**2 - a**2) * np.sin(theta) * np.cos(theta)) * x * y
        + ((a * np.cos(theta)) ** 2 + (b * np.sin(theta)) ** 2) * y**2
    ) < a**2 * b**2
    binary_map = binary_map.reshape(resolution, resolution).astype(int)

    contours, hierarchy = cv2.findContours(binary_map, mode=cv2.RETR_FLOODFILL, method=cv2.CHAIN_APPROX_SIMPLE)
    parent_indices = np.flatnonzero(hierarchy[..., 3] == -1)
    contours = [np.squeeze(contours[i]) for i in parent_indices]

    final_contour = contours[0] if not centered else contours[0] - resolution / 2

    return final_contour.astype(np.float32)


def area_of_circular_segment(circle_radius: float, delta_height: float) -> float:
    """Compute the area of a circular segment (see source for definition).

    Source: https://en.wikipedia.org/wiki/Circular_segment

    Args:
        circle_radius (float): Radius of the circle (R).
        delta_height (float): distance between the center of the segment and the base of the secant, i.e. apothem (d).

    Returns:
        float: area of the circular segment
    """
    if delta_height > circle_radius:
        return 0.0
    area = circle_radius**2 * np.arccos(delta_height / circle_radius) - delta_height * np.sqrt(
        circle_radius**2 - delta_height**2
    )
    return area


def rotated_asymetric_rectangle(
    center_x: float, center_y: float, semi_width: float, upper_height: float, lower_height: float, angle: float
) -> np.ndarray:
    """Compute a rotated rectangle with different upper and lower semi-heights.

    Args:
        center_x (float): X coordinates of the center of the asymetric rectangle.
        center_y (float): Y coordinates of the center of the asymetric rectangle
        semi_width (float): half of the recantgle width.
        upper_height (float): distance from the center of the rectangle to the upper edge.
        lower_height (float): distance from the center of the rectangle to the lower edge.
        angle (float): angle of rotation in radians.

    Returns:
        np.ndarray: rotated rectangle array.
    """
    return np.array(
        [
            [
                center_x + semi_width * np.cos(angle) + upper_height * np.cos(np.pi / 2 + angle),
                center_y + semi_width * np.sin(angle) + upper_height * np.sin(np.pi / 2 + angle),
            ],
            [
                center_x + semi_width * np.cos(angle) - lower_height * np.cos(np.pi / 2 + angle),
                center_y + semi_width * np.sin(angle) - lower_height * np.sin(np.pi / 2 + angle),
            ],
            [
                center_x - semi_width * np.cos(angle) - lower_height * np.cos(np.pi / 2 + angle),
                center_y - semi_width * np.sin(angle) - lower_height * np.sin(np.pi / 2 + angle),
            ],
            [
                center_x - semi_width * np.cos(angle) + upper_height * np.cos(np.pi / 2 + angle),
                center_y - semi_width * np.sin(angle) + upper_height * np.sin(np.pi / 2 + angle),
            ],
        ]
    )


def compare_iris_pipeline_metadata_output(metadata_1: Dict[str, Any], metadata_2: Dict[str, Any]) -> None:
    """Compare two IRISPipeline outputs

    Args:
        metadata_1 (Dict[str, Any]): pipeline's metadata output 1.
        metadata_2 (Dict[str, Any]): pipeline's metadata output 2.
    """
    assert metadata_2["image_size"] == metadata_1["image_size"]
    assert metadata_2["eye_side"] == metadata_1["eye_side"]
    np.testing.assert_almost_equal(
        metadata_2["eye_centers"]["pupil_center"],
        metadata_1["eye_centers"]["pupil_center"],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        metadata_2["eye_centers"]["iris_center"],
        metadata_1["eye_centers"]["iris_center"],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        list(metadata_2["pupil_to_iris_property"].values()),
        list(metadata_1["pupil_to_iris_property"].values()),
        decimal=6,
    )
    np.testing.assert_almost_equal(
        metadata_2["offgaze_score"],
        metadata_1["offgaze_score"],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        metadata_2["eye_orientation"],
        metadata_1["eye_orientation"],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        metadata_2["occlusion90"],
        metadata_1["occlusion90"],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        metadata_2["occlusion30"],
        metadata_1["occlusion30"],
        decimal=6,
    )
    np.testing.assert_almost_equal(
        [
            metadata_2["iris_bbox"]["x_min"],
            metadata_2["iris_bbox"]["x_max"],
            metadata_2["iris_bbox"]["y_min"],
            metadata_2["iris_bbox"]["y_max"],
        ],
        [
            metadata_1["iris_bbox"]["x_min"],
            metadata_1["iris_bbox"]["x_max"],
            metadata_1["iris_bbox"]["y_min"],
            metadata_1["iris_bbox"]["y_max"],
        ],
        decimal=6,
    )


def compare_iris_pipeline_template_output(iris_template_1: Dict[str, Any], iris_template_2: Dict[str, Any]) -> None:
    """Compare two IRISPipeline template outputs

    Args:
        iris_template_1 (Dict[str, Any]): pipeline's iris template output 1.
        iris_template_2 (Dict[str, Any]): pipeline's iris template output 2.
    """
    assert iris_template_2["iris_codes"] == iris_template_1["iris_codes"]
    assert iris_template_2["mask_codes"] == iris_template_1["mask_codes"]


def compare_iris_pipeline_error_output(error_dict_1: Dict[str, str], error_dict_2: Dict[str, str]) -> None:
    """Compare two IRISPipeline error outputs

    Args:
        error_dict_1 (Dict[str, str]): pipeline's error output 1.
        error_dict_2 (Dict[str, str]): pipeline's error output 2.
    """
    assert (error_dict_1 is None) == (error_dict_2 is None)
    if error_dict_1 is not None:
        assert error_dict_1["error_type"] == error_dict_2["error_type"]
        assert error_dict_1["traceback"] == error_dict_2["traceback"]
        assert error_dict_1["message"] == error_dict_2["message"]


def compare_iris_pipeline_outputs(pipeline_output_1: Dict[str, Any], pipeline_output_2: Dict[str, Any]):
    """Compare two IRISPipeline outputs for the Orb

    Args:
        pipeline_output_1 (Dict[str, Any]): pipeline output 1.
        pipeline_output_2 (Dict[str, Any]): pipeline output 2.
    """
    compare_iris_pipeline_template_output(pipeline_output_1["iris_template"], pipeline_output_2["iris_template"])
    compare_iris_pipeline_metadata_output(pipeline_output_1["metadata"], pipeline_output_2["metadata"])
    compare_iris_pipeline_error_output(pipeline_output_1["error"], pipeline_output_2["error"])


def compare_debug_pipeline_outputs(pipeline_output_1: Dict[str, Any], pipeline_output_2: Dict[str, Any]):
    """Compare two IRISPipeline outputs for debugging.

    Args:
        pipeline_output_1 (Dict[str, Any]): pipeline output 1.
        pipeline_output_2 (Dict[str, Any]): pipeline output 2.
    """
    compare_iris_pipeline_template_output(pipeline_output_1["iris_template"], pipeline_output_2["iris_template"])
    compare_iris_pipeline_metadata_output(pipeline_output_1["metadata"], pipeline_output_2["metadata"])

    # Debug-specific intermediary outputs
    to_test = {
        "normalized_iris": ["normalized_image", "normalized_mask"],
        "iris_response": ["iris_responses", "mask_responses"],
        "extrapolated_polygons": ["pupil", "iris", "eyeball"],
        "landmarks": ["pupil", "iris", "eyeball"],
    }
    for key, values in to_test.items():
        for value in values:
            np.testing.assert_almost_equal(
                pipeline_output_1[key][value],
                pipeline_output_2[key][value],
                decimal=4,
            )
    np.testing.assert_almost_equal(
        pipeline_output_1["segmentation_map"]["predictions"],
        pipeline_output_2["segmentation_map"]["predictions"],
        decimal=4,
    )
