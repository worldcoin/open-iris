from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

from pydantic import validator

from iris.io.class_configs import ImmutableModel
from iris.orchestration.validators import pipeline_metadata_version_check


class PipelineMetadata(ImmutableModel):
    """Data holder for input config's metadata."""

    pipeline_name: str
    iris_version: str

    _version_check = validator("iris_version", allow_reuse=True)(pipeline_metadata_version_check)


class PipelineValue(ImmutableModel):
    """Data holder for pipeline value that flows through the system with optional index to specify value this holder refers to if value is of Iterable type."""

    name: str
    index: Optional[int]


class PipelineInput(PipelineValue):
    """Data holder for the reference to an input node."""

    source_node: Union[str, List[Union[str, PipelineValue]]]


class PipelineClass(ImmutableModel):
    """Data holder for the reference to any class: Algorithm, Callback, etc."""

    class_name: str
    params: Dict[
        str,
        Union[
            int,
            float,
            str,
            PipelineClass,
            Tuple[int, float, str, PipelineClass],
            List[Union[int, float, str, PipelineClass]],
        ],
    ]


class PipelineNode(ImmutableModel):
    """Data holder for one node in a declared pipeline."""

    name: str
    algorithm: PipelineClass
    inputs: List[PipelineInput]
    callbacks: Optional[List[PipelineClass]]
    seed: Optional[str] = None
