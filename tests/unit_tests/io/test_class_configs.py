import sys
from typing import Any, Dict

import pytest
from pydantic import Field, ValidationError

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm, ImmutableModel, instantiate_class_from_name


class ConcreteImmutableModel(ImmutableModel):
    """A concrete implementation of ImmutableModel with parameters"""

    my_param_1: int = Field(..., gt=0)
    my_param_2: str


@pytest.mark.parametrize(
    "parameters",
    [
        ({"my_param_1": 3, "my_param_2": "toto"}),
        ({"my_param_1": 3, "my_param_2": "3.7"}),
    ],
)
def test_immutable_model_constructor(parameters: Dict[str, Any]) -> None:
    cim = ConcreteImmutableModel(**parameters)

    for key, value in parameters.items():
        assert getattr(cim, key) == value


@pytest.mark.parametrize(
    "parameters",
    [
        ({"my_param_1": -4, "my_param_2": "toto"}),
    ],
    ids=["pydantic checks"],
)
def test_immutable_model_constructor_raises_exception(parameters: Dict) -> None:
    with pytest.raises((ValidationError, TypeError)):
        _ = ConcreteImmutableModel(**parameters)


@pytest.mark.parametrize(
    "parameters,new_parameters",
    [
        pytest.param(
            {"my_param_1": 3, "my_param_2": "toto"},
            {"my_param_1": 6, "my_param_2": "not toto"},
        ),
    ],
    ids=["regular"],
)
def test_immutability_of_immutable_model(parameters: Dict[str, Any], new_parameters: Dict[str, Any]) -> None:
    immutable_obj = ConcreteImmutableModel(**parameters)

    with pytest.raises(TypeError):
        for key, value in new_parameters.items():
            setattr(immutable_obj, key, value)


class MockDummyValidationAlgorithm(Callback):
    CORRECT_MSG = "Worldcoin AI is the best"
    ERROR_MSG = "Incorrect msg returned!"

    def on_execute_end(self, result: str) -> None:
        if result != self.CORRECT_MSG:
            raise RuntimeError(MockDummyValidationAlgorithm.ERROR_MSG)


class MockParametrizedModelWithCallback(Algorithm):
    class Parameters(Algorithm.Parameters):
        ret_msg: str

    __parameters_type__ = Parameters

    def __init__(self, ret_msg: str = "Worldcoin AI is the best") -> None:
        super().__init__(ret_msg=ret_msg, callbacks=[MockDummyValidationAlgorithm()])

    def run(self) -> str:
        return self.params.ret_msg


def test_parametrized_model_validation_hook_not_raising_an_error() -> None:
    mock_model = MockParametrizedModelWithCallback()

    result = mock_model.execute()

    assert result == mock_model.params.ret_msg


def test_parametrized_model_validation_hook_raising_an_error() -> None:
    mock_model = MockParametrizedModelWithCallback(ret_msg="Worldcoin AI isn't the best")

    with pytest.raises(RuntimeError) as err:
        _ = mock_model.execute()

    assert str(err.value) == MockDummyValidationAlgorithm.ERROR_MSG


def test_from_name_instantiates_algorithm_subclass():
    class DummyAlgorithm(Algorithm):
        class Parameters(Algorithm.Parameters):
            foo: int

        __parameters_type__ = Parameters

        def run(self):
            return self.params.foo

    # Register DummyAlgorithm in the module namespace for pydoc.locate
    current_module = sys.modules[__name__]
    setattr(current_module, "DummyAlgorithm", DummyAlgorithm)

    # Should instantiate and run correctly
    instance = instantiate_class_from_name(f"{__name__}.DummyAlgorithm", {"foo": 42})
    assert isinstance(instance, DummyAlgorithm)
    assert instance.run() == 42


def test_from_name_instantiates_non_algorithm_class():
    class NonAlgorithm:
        def __init__(self, bar):
            self.bar = bar

    current_module = sys.modules[__name__]
    setattr(current_module, "NonAlgorithm", NonAlgorithm)

    instance = instantiate_class_from_name(f"{__name__}.NonAlgorithm", {"bar": "baz"})
    assert isinstance(instance, NonAlgorithm)
    assert instance.bar == "baz"


def test_from_name_raises_on_missing_class():
    with pytest.raises(ValueError) as excinfo:
        instantiate_class_from_name("nonexistent.module.ClassName", {})
    assert "Could not locate class" in str(excinfo.value)
