import abc
from typing import Any

import numpy as np

from iris.io.class_configs import Algorithm
from iris.io.errors import ImageFilterError


class ImageFilter(Algorithm):
    """Image filter abstract class."""

    class Parameters(Algorithm.Parameters):
        """Default ImageFilter parameters."""

        pass

    __parameters_type__ = Parameters

    def __init__(self, **kwargs: Any) -> None:
        """Init function."""
        super().__init__(**kwargs)
        self.__kernel_values = self.compute_kernel_values()

    @property
    def kernel_values(self) -> np.ndarray:
        """Get kernel values.

        Returns:
            np.ndarray: Filter kernel values.
        """
        return self.__kernel_values

    @kernel_values.setter
    def kernel_values(self, value: Any) -> None:
        """Prevent overwriting generated kernel values.

        Args:
            value (Any): New kernel values.

        Raises:
            ImageFilterError: Raised always since overwriting is forbidden.
        """
        raise ImageFilterError("ImageFilter kernel_values are immutable.")

    @abc.abstractmethod
    def compute_kernel_values(self) -> np.ndarray:
        """Compute values of filter kernel.

        Returns:
            np.ndarray: Computed kernel values.
        """
        pass
