from typing import Any, Callable, Dict, List

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.class_configs import Algorithm, ImmutableModel
from iris.orchestration.pipeline_dataclasses import PipelineNode


class Environment(ImmutableModel):
    """Data holder for the pipeline environment properties.

    call_trace_initialiser is responsible for initialising the PipelineCallTraceStorage instance in the pipeline.

    pipeline_output_builder is responsible for building the pipeline output from the call_trace, which kept all intermediary results so far.

    error_manager is responsible for the pipeline's behaviour in case of an exception

    disabled_qa stores a list of Algorithm and/or Callbacks types to be disabled.
    """

    call_trace_initialiser: Callable[[Dict[str, Algorithm], List[PipelineNode]], PipelineCallTraceStorage]
    pipeline_output_builder: Callable[[PipelineCallTraceStorage], Any]
    error_manager: Callable[[PipelineCallTraceStorage, Exception], None]
    disabled_qa: List[type] = []
