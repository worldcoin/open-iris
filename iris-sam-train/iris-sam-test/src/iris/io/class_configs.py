import abc
from copy import deepcopy
from typing import Any, List

import pydantic
from pydantic import Extra

from iris.callbacks.callback_interface import Callback


class ImmutableModel(pydantic.BaseModel):
    """Specifies configurations for validating classes which objects should be immutable."""

    class Config:
        """Configuration options for classes which objects are meant to be immutable."""

        arbitrary_types_allowed = True
        allow_mutation = False
        validate_all = True
        smart_union = True
        extra = Extra.forbid

    def serialize(self) -> Any:
        """Serialize the object. By defaults, this method raises a RuntimeError to notify the user that the method wasn't implemented.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(f"{self.__class__.__name__}.serialize not implemented!")

    @staticmethod
    def deserialize(self) -> Any:
        """Deserialize the object. By defaults, this method raises a RuntimeError to notify the user that the method wasn't implemented.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(f"{self.__class__.__name__}.deserialize not implemented!")


class Algorithm(abc.ABC):
    """Base class of every node of the iris recognition pipeline."""

    class Parameters(ImmutableModel):
        """Default parameters."""

        pass

    __parameters_type__ = Parameters

    def __init__(self, **kwargs: Any) -> None:
        """Init function."""
        self._callbacks: List[Callback] = []

        if "callbacks" in kwargs.keys():
            self._callbacks = deepcopy(kwargs["callbacks"])
            del kwargs["callbacks"]

        self.params = self.__parameters_type__(**kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Make an object a functor.

        Returns:
            Any: Object specified by an interface.
        """
        return self.execute(*args, **kwargs)

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        """Execute method and wrapped with hooks if such are specified.

        Returns:
            Any: Object specified by an interface.
        """
        for callback_func in self._callbacks:
            callback_func.on_execute_start(*args, **kwargs)

        result = self.run(*args, **kwargs)

        for callback_func in self._callbacks:
            callback_func.on_execute_end(result)

        return result

    def run(self, *args: Any, **kwargs: Any) -> Any:
        """Implement method design pattern. Not overwritten by subclass will raise an error.

        Raises:
            NotImplementedError: Raised if subclass doesn't implement `run` method.

        Returns:
            Any: Return value by concrate implementation of the `run` method.
        """
        raise NotImplementedError(f"{self.__class__.__name__}.run method not implemented!")
