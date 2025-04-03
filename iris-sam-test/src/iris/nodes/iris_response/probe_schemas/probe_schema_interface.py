import abc
from typing import Any, Tuple

import numpy as np

from iris.io.class_configs import Algorithm
from iris.io.errors import ProbeSchemaError


class ProbeSchema(Algorithm):
    """Probe schema abstract class."""

    class ProbeSchemaParameters(Algorithm.Parameters):
        """Default ProbeSchema parameters."""

    __parameters_type__ = ProbeSchemaParameters

    def __init__(self, **kwargs: Any) -> None:
        """Init function."""
        super().__init__(**kwargs)
        self.__rhos, self.__phis = self.generate_schema()

    @property
    def rhos(self) -> np.ndarray:
        """Get rhos' position values.

        Returns:
            np.ndarray: rhos' position values.
        """
        return self.__rhos

    @rhos.setter
    def rhos(self, value: Any) -> None:
        """Prevent overwriting generated rhos' positions values.

        Args:
            value (Any): New rhos' position values.

        Raises:
            ProbeSchemaError: Raised always since overwriting is forbidden.
        """
        raise ProbeSchemaError("ProbeSchema rhos values are immutable.")

    @property
    def phis(self) -> np.ndarray:
        """Get phis' position values.

        Returns:
            np.ndarray: phis' position values.
        """
        return self.__phis

    @phis.setter
    def phis(self, value: Any) -> None:
        """Prevent overwriting generated phis' positions values.

        Args:
            value (Any): New phis' position values.

        Raises:
            ProbeSchemaError: Raised always since overwriting is forbidden.
        """
        raise ProbeSchemaError("ProbeSchema phis values are immutable.")

    @abc.abstractmethod
    def generate_schema(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate rhos' and phis' positions values.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Tuple with generated schema (rhos, phis).
        """
        pass
