from typing import Any, Dict, List

import numpy as np
import pytest

import iris.io.validators as pydantic_v


class MockClass:
    pass


class MockField:
    name = "mock_field"


def test_is_binary() -> None:
    v = np.array([0, 0, 1, 1, 0]).astype(bool)

    _ = pydantic_v.is_binary(MockClass, v, MockField)


@pytest.mark.parametrize(
    "v",
    [
        pytest.param(np.array([0, 0, 1, 1, 0])),
        pytest.param(np.array([1, 5, 29])),
    ],
    ids=["binary not bool", "not binary"],
)
def test_is_binary_raises_an_exception(v: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = pydantic_v.is_binary(MockClass, v, MockField)


def test_is_list_of_points() -> None:
    v = np.random.random((30, 2))

    _ = pydantic_v.is_list_of_points(MockClass, v, MockField)


@pytest.mark.parametrize(
    "v",
    [
        pytest.param(np.random.random((10, 10, 10))),
        pytest.param(np.random.random((30, 3))),
    ],
    ids=["3-dimensional", "list of 3d points"],
)
def test_is_list_of_points_raises_an_exception(v: np.ndarray) -> None:
    with pytest.raises(ValueError):
        _ = pydantic_v.is_list_of_points(MockClass, v, MockField)


@pytest.mark.parametrize(
    "v",
    [
        ["not empty"],
        [None],
    ],
    ids=["not empty 1", "not empty 2"],
)
def test_is_not_empty(v: List[Any]) -> None:
    pydantic_v.is_not_empty(MockClass, v, MockField)


def test_is_not_empty_raises_an_exception() -> None:
    v: List[Any] = []

    with pytest.raises(ValueError):
        _ = pydantic_v.is_not_empty(MockClass, v, MockField)


@pytest.mark.parametrize(
    "v",
    [
        (0),
        (1),
        (0.0),
        (-0.0),
        (1.0),
        (np.array([1, 2, 3])),
        (np.array([1.0, 2.0, 3.0])),
        (np.array([[1.0, 2.0, 3.0], [4, 5.0, 8.5]])),
    ],
    ids=["0", "int >0", "0.0", "-0.0", "float>0", "int array", "float array", "2d array"],
)
def test_are_all_positive(v: Any) -> None:
    pydantic_v.are_all_positive(MockClass, v, MockField)


@pytest.mark.parametrize(
    "v",
    [(-1), (-np.pi), (np.array([-1, 2, 3])), (np.array([1.0, -2.4, 0])), (np.array([[1.0, 2.0, -3.0], [4, 5.0, 8.5]]))],
    ids=[
        "negative int",
        "negative float array",
        "negative int array",
        "float array with negative",
        "2d array with negative",
    ],
)
def test_are_all_positive_raises_an_exception(v: Any) -> None:
    with pytest.raises(ValueError):
        _ = pydantic_v.are_all_positive(MockClass, v, MockField)


@pytest.mark.parametrize(
    "nb_dimensions,v",
    [
        (0, np.array([])),
        (1, np.array([0, 1, 2, 3])),
        (3, np.zeros((2, 2, 2))),
    ],
    ids=[
        "expected dim 0, got dim 0",
        "expected dim 1, got dim 1",
        "expected dim 3, got dim 3",
    ],
)
def test_is_array_n_dimensions(nb_dimensions: int, v: np.ndarray) -> None:
    val = pydantic_v.is_array_n_dimensions(nb_dimensions)
    val(MockClass, v, MockField)


@pytest.mark.parametrize(
    "nb_dimensions,v",
    [
        pytest.param(0, np.array([1])),
        pytest.param(2, np.array([1])),
        pytest.param(2, np.zeros((2, 2, 2))),
    ],
    ids=[
        "expected dim 0, got dim 1",
        "expected dim 2, got dim 1",
        "expected dim 2, got dim 3",
    ],
)
def test_is_array_n_dimensions_raises_an_exception(nb_dimensions: int, v: np.ndarray) -> None:
    with pytest.raises(ValueError):
        val = pydantic_v.is_array_n_dimensions(nb_dimensions)
        val(MockClass, v, MockField)


@pytest.mark.parametrize(
    "values",
    [
        {"x_min": 14, "x_max": 2857.0, "y_min": 142.857, "y_max": 571.428},
        {"x_min": 0, "x_max": 100.0, "y_min": 0.0, "y_max": 314.15},
        {"x_min": -10, "x_max": 20.0, "y_min": 14.2857, "y_max": 142857},
        {"x_min": -20, "x_max": -10.0, "y_min": -142857, "y_max": -14.2857},
    ],
    ids=["any bbox", "bbox with x/y_min at 0", "negative values", "all negative values"],
)
def test_is_valid_bbox(values: Dict[str, float]) -> None:
    pydantic_v.is_valid_bbox(MockClass, values)


@pytest.mark.parametrize(
    "values",
    [
        {"x_min": 1000000, "x_max": 1.0, "y_min": 0.0, "y_max": 10000},
        {"x_min": 10, "x_max": 20, "y_min": 8, "y_max": -10},
        {"x_min": 1, "x_max": 10000.0, "y_min": 100000, "y_max": 0.0},
        {"x_min": 100000, "x_max": 1.0, "y_min": 100000.0, "y_max": 0},
        {"x_min": 142857, "x_max": 142857.0, "y_min": 142.0, "y_max": 857},
        {"x_min": 10, "x_max": 20.0, "y_min": 142857.0, "y_max": 142857},
    ],
    ids=[
        "x_min > x_max",
        "negative y_max positive y_min",
        "y_min > y_max",
        "x_min > x_max and y_min > y_max",
        "x_min = x_max",
        "y_min = y_max",
    ],
)
def test_is_valid_bbox_raises_error(values: Dict[str, float]) -> None:
    with pytest.raises(ValueError):
        pydantic_v.is_valid_bbox(MockClass, values)


@pytest.mark.parametrize(
    "values",
    [
        {"field1": [], "field2": []},
        {"field1": ["4"], "field2": [8]},
        {"field1": [[1, 2], [1, 2, 3, 4], [1]], "field2": [[1], [], [1, 2, 3]]},
    ],
    ids=["both empty", "both len 1", "both len 3"],
)
def test_are_length_equal(values: Dict[str, np.ndarray]) -> None:
    val = pydantic_v.are_lengths_equal(field1="field1", field2="field2")
    val(MockClass, values)


@pytest.mark.parametrize(
    "values",
    [
        pytest.param({"field1": [1, 2], "field2": [1, 2, 3, 4]}),
        pytest.param({"field1": [1, 2], "field2": []}),
    ],
    ids=["different length", "one empty one not"],
)
def test_are_length_equal_raises_an_exception(values: Dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError):
        val = pydantic_v.are_lengths_equal(field1="field1", field2="field2")
        val(MockClass, values)


@pytest.mark.parametrize(
    "values",
    [
        {"field1": np.array([]), "field2": np.array([])},
        {"field1": np.zeros((2, 3)), "field2": np.ones((2, 3))},
    ],
    ids=["both empty", "both shape (2, 3)"],
)
def test_are_shapes_equal(values: Dict[str, np.ndarray]) -> None:
    val = pydantic_v.are_shapes_equal(field1="field1", field2="field2")
    val(MockClass, values)


@pytest.mark.parametrize(
    "values",
    [
        ({"field1": np.zeros((2, 3)), "field2": np.array([])}),
        ({"field1": np.random.random((5, 17, 6)), "field2": np.random.random((6, 17, 5))}),
    ],
    ids=["one empty one not", "different length"],
)
def test_are_shapes_equal_raises_an_exception(values: Dict[str, np.ndarray]) -> None:
    with pytest.raises(ValueError):
        val = pydantic_v.are_shapes_equal(field1="field1", field2="field2")
        val(MockClass, values)


@pytest.mark.parametrize(
    "values",
    [
        {"field1": [], "field2": []},
        {"field1": [np.zeros((2, 3)) for _ in range(5)], "field2": [np.ones((2, 3)) for _ in range(5)]},
    ],
    ids=["both empty", "both same number of shapes (2, 3)"],
)
def test_are_all_shapes_equal(values: Dict[str, List[np.ndarray]]) -> None:
    val = pydantic_v.are_all_shapes_equal(field1="field1", field2="field2")
    val(MockClass, values)


@pytest.mark.parametrize(
    "values",
    [
        ({"field1": [np.zeros((2, 3))], "field2": np.array([])}),
        ({"field1": [np.zeros((2, 3)) for _ in range(5)], "field2": [np.ones((2, 3)) for _ in range(3)]}),
    ],
    ids=["one empty one not", "different length"],
)
def test_are_all_shapes_equal_raises_an_exception(values: Dict[str, List[np.ndarray]]) -> None:
    with pytest.raises(ValueError):
        val = pydantic_v.are_all_shapes_equal(field1="field1", field2="field2")
        val(MockClass, values)
