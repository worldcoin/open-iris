import abc
from typing import Any, List

from iris.io.class_configs import ImmutableModel
from iris.io.dataclasses import IrisTemplate
from pydantic import conint


class Matcher(abc.ABC):
    """Parent Abstract class for 1-to-1 matchers."""

    class Parameters(ImmutableModel):
        """IrisMatcherParameters parameters."""

        rotation_shift: conint(ge=0, strict=True)

    __parameters_type__ = Parameters

    def __init__(self, **kwargs) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int = 15): rotation allowed in matching, converted to columns. Defaults to 15.
        """
        self.params = self.__parameters_type__(**kwargs)

    @abc.abstractmethod
    def run(self, template_probe: IrisTemplate, template_gallery: IrisTemplate) -> float:
        """Match iris templates using Hamming distance.

        Args:
            template_probe (IrisTemplate): Iris template from probe.
            template_gallery (IrisTemplate): Iris template from gallery.

        Returns:
            float: matching distance.
        """
        pass


class BatchMatcher(abc.ABC):
    """Parent Abstract class for 1-to-N matchers."""

    class Parameters(ImmutableModel):
        """IrisMatcherParameters parameters."""

        rotation_shift: conint(ge=0, strict=True)

    __parameters_type__ = Parameters

    def __init__(self, **kwargs: Any) -> None:
        """Assign parameters.

        Args:
            rotation_shift (int = 15): rotation allowed in matching, converted to columns. Defaults to 15.
        """
        self.params = self.__parameters_type__(**kwargs)

    @abc.abstractmethod
    def intra_gallery(self, template_gallery: List[IrisTemplate]) -> List[List[float]]:
        """Match iris templates using Hamming distance.

        Args:
            template_gallery (List[IrisTemplate]): Iris template gallery.

        Returns:
            List[List[float]]: matching distances.
        """
        pass

    @abc.abstractmethod
    def gallery_to_gallery(
        self, template_gallery_1: List[IrisTemplate], template_gallery_2: List[IrisTemplate]
    ) -> List[List[float]]:
        """Match iris templates using Hamming distance.

        Args:
            template_gallery_1 (List[IrisTemplate]): Iris template gallery.
            template_gallery_2 (List[IrisTemplate]): Iris template gallery.

        Returns:
            List[List[float]]: matching distances.
        """
        pass
