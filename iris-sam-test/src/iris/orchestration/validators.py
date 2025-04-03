from collections import Counter
from typing import Any, List

from pydantic import fields

from iris._version import __version__
from iris.io.errors import IRISPipelineError


def pipeline_config_duplicate_node_name_check(cls: type, v: List[Any], field: fields.ModelField) -> List[Any]:
    """Check if all pipeline nodes have distinct names.

    Args:
        cls (type): Class type.
        v (List[Any]): Value to check.
        field (fields.ModelField): Field descriptor.

    Raises:
        IRISPipelineError: Raised if pipeline nodes aren't unique.

    Returns:
        List[Any]: `v` sent for further processing.
    """
    node_names = [node.name for node in v]

    if len(set(node_names)) != len(node_names):
        raise IRISPipelineError(f"Pipeline node name must be unique. Received {dict(Counter(node_names))}")

    return v


def pipeline_metadata_version_check(cls: type, v: str, field: fields.ModelField) -> str:
    """Check if the version provided in the input config matches the current iris.__version__."""
    if v != __version__:
        raise IRISPipelineError(
            f"Wrong config version. Cannot initialise IRISPipeline version {__version__} on a config file "
            f"version {v}"
        )

    return v
