import re
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
from pydantic import fields

from iris.io.errors import IRISPipelineError

# ----- validators -----


def is_odd(cls: type, v: int, field: fields.ModelField) -> int:
    """Check that kernel size are odd numbers.

    Args:
        cls (type): Class type.
        v (int): Value to check.
        field (fields.ModelField): Field descriptor.

    Raises:
        ValueError: Exception raised if number isn't odd.

    Returns:
        int: `v` sent for further processing.
    """
    if (v % 2) == 0:
        raise ValueError(f"{cls.__name__}: {field.name} must be odd numbers.")

    return v


def is_uint8(cls: type, v: np.ndarray, field: fields.ModelField) -> np.ndarray:
    """Check if np array contains only uint8 values."""
    values_check = not (np.all(v >= 0) and np.all(v <= 255))
    if values_check or v.dtype != np.uint8:
        raise ValueError(f"{cls.__name__}: {field.name} must be of uint8 type. Received {v.dtype}")
    return v


def is_binary(cls: type, v: np.ndarray, field: fields.ModelField) -> np.ndarray:
    """Check if array has only boolean values, i.e. is binary.

    Args:
        cls (type): Class type.
        v (np.ndarray): Value to check.
        field (fields.ModelField): Field descriptor.

    Raises:
        ValueError: Exception raised if array doesn't contain bool datatypes.

    Returns:
        np.ndarray: `v` sent for further processing.
    """
    if v.dtype != np.dtype("bool"):
        raise ValueError(f"{cls.__name__}: {field.name} must be binary. got dtype {v.dtype}")

    return v


def is_list_of_points(cls: type, v: np.ndarray, field: fields.ModelField) -> np.ndarray:
    """Check if np.ndarray has shape (_, 2).

    Args:
        cls (type): Class type.
        v (np.ndarray): Value to check.
        field (fields.ModelField): Field descriptor.

    Raises:
        ValueError: Exception raised if array doesn't contain 2D points.

    Returns:
        np.ndarray: `v` sent for further processing.
    """
    if len(v.shape) != 2 or v.shape[1] != 2:
        raise ValueError(f"{cls.__name__}: {field.name} must have shape (_, 2).")

    return v


def is_not_empty(cls: type, v: List[Any], field: fields.ModelField) -> List[Any]:
    """Check that both inputs are not empty.

    Args:
        cls (type): Class type.
        v (List[Any]): Value to check.
        field (fields.ModelField): Field descriptor.

    Raises:
        ValueError: Exception raised if list is empty.

    Returns:
        List[Any]: `v` sent for further processing.
    """
    if len(v) == 0:
        raise ValueError(f"{cls.__name__}: {field.name} list cannot be empty.")

    return v


def is_not_zero_sum(cls: type, v: Any, field: fields.ModelField) -> Any:
    """Check that both inputs are not empty.

    Args:
        cls (type): Class type.
        v (Any): Value to check.
        field (fields.ModelField): Field descriptor.

    Raises:
        ValueError: Raised if v doesn't sum to 0.

    Returns:
        Any: `v` sent for further processing.
    """
    if np.sum(v) == 0:
        raise ValueError(f"{cls.__name__}: {field.name} sum cannot be zero.")

    return v


def are_all_positive(cls: type, v: Any, field: fields.ModelField) -> Any:
    """Check that all values are positive.

    Args:
        cls (type): Class type.
        v (Any): Value to check.
        field (fields.ModelField): Field descriptor.

    Raises:
        ValueError: Raise if not all values in are positive.

    Returns:
        Any: `v` sent for further processing.
    """
    if isinstance(v, Iterable):
        if not np.array([value >= 0 for value in v]).all():
            raise ValueError(f"{cls.__name__}: all {field.name} must be positive. Received {v}")
    elif v < 0.0:
        raise ValueError(f"{cls.__name__}: {field.name} must be positive. Received {v}")

    return v


def iris_code_version_check(cls: type, v: str, field: fields.ModelField) -> str:
    """Check if the version provided in the input config matches the current iris.__version__."""
    if not re.match(r"v[\d]+\.[\d]+$", v):
        raise IRISPipelineError(f"Wrong iris code version. Expected standard version nuber, received {v}")
    return v


def to_dtype_float32(cls: type, v: np.ndarray, field: fields.ModelField) -> np.ndarray:
    """Convert input np.ndarray to dtype np.float32.

    Args:
        cls (type): Class type.
        v (np.ndarray): Value to convert
        field (fields.ModelField): Field descriptor.

    Returns:
        np.ndarray: `v` sent for further processing.
    """
    return v.astype(np.float32)


# ----- root_validators -----


def is_valid_bbox(cls: type, values: Dict[str, float]) -> Dict[str, float]:
    """Check that the bounding box is valid."""
    if values["x_min"] >= values["x_max"] or values["y_min"] >= values["y_max"]:
        raise ValueError(
            f'{cls.__name__}: invalid bbox. x_min={values["x_min"]}, x_max={values["x_max"]},'
            f' y_min={values["y_min"]}, y_max={values["y_max"]}'
        )

    return values


# ----- parametrized validators -----


def is_array_n_dimensions(nb_dimensions: int) -> Callable:
    """Create a pydantic validator checking if an array is n-dimensional.

    Args:
        nb_dimensions (int): number of dimensions the array must have

    Returns:
        Callable: the validator.
    """

    def validator(cls: type, v: np.ndarray, field: fields.ModelField) -> np.ndarray:
        """Check if the array has the right number of dimensions."""
        if len(v.shape) != nb_dimensions and (v.shape != (0,) or nb_dimensions != 0):
            raise ValueError(
                f"{cls.__name__}: wrong number of dimensions for {field.name}. "
                f"Expected {nb_dimensions}, got {len(v.shape)}"
            )

        return v

    return validator


# ----- parametrized root_validators -----


def are_lengths_equal(field1: str, field2: str) -> Callable:
    """Create a pydantic validator checking if the two fields have the same length.

    Args:
        field1 (str): name of the first field
        field2 (str): name of the first field

    Returns:
        Callable: the validator.
    """

    def __root_validator(cls: type, values: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        """Check if len(field1) equals len(field2)."""
        if len(values[field1]) != len(values[field2]):
            raise ValueError(
                f"{cls.__name__}: {field1} and {field2} length mismatch, "
                f"resp. {len(values[field1])} and {len(values[field2])}"
            )

        return values

    return __root_validator


def are_shapes_equal(field1: str, field2: str) -> Callable:
    """Create a pydantic validator checking if the two fields have the same shape.

    Args:
        field1 (str): name of the first field
        field2 (str): name of the first field

    Returns:
        Callable: the validator.
    """

    def __root_validator(cls: type, values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Check if field1.shape equals field2.shape."""
        if values[field1].shape != values[field2].shape:
            raise ValueError(f"{cls.__name__}: {field1} and {field2} shape mismatch.")
        return values

    return __root_validator


def are_all_shapes_equal(field1: str, field2: str) -> Callable:
    """Create a pydantic validator checking if two lists of array have the same shape per element.

    This function creates a pydantic validator for two lists of np.ndarrays which checks if they have the same length,
    and if all of their element have the same shape one by one.

    Args:
        field1 (str): name of the first field
        field2 (str): name of the first field

    Returns:
        Callable: the validator.
    """

    def __root_validator(cls: type, values: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Check if len(field1) equals len(field2) and if every element have the same shape."""
        shapes_field_1 = [element.shape for element in values[field1]]
        shapes_field_2 = [element.shape for element in values[field2]]

        if len(values[field1]) != len(values[field2]) or shapes_field_1 != shapes_field_2:
            raise ValueError(
                f"{cls.__name__}: {field1} and {field2} shape mismatch, resp. {shapes_field_1} and {shapes_field_2}."
            )

        return values

    return __root_validator
