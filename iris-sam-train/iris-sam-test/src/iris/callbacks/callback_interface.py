import abc
from typing import Any


class Callback(abc.ABC):
    """Base class of the Callback API."""

    def on_execute_start(self, *args: Any, **kwargs: Any) -> None:
        """Execute this method called before node execute method."""
        pass

    def on_execute_end(self, result: Any) -> None:
        """Execute this method called after node execute method.

        Args:
            result (Any): execute method output.
        """
        pass
