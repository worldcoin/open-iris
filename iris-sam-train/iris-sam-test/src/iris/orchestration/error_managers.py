from iris.callbacks.pipeline_trace import PipelineCallTraceStorage


def raise_error_manager(call_trace: PipelineCallTraceStorage, exception: Exception) -> None:
    """Error manager for the Orb.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call results storage.
        exception (Exception): Exception raised during the pipeline call.

    Raises:
        Exception: Reraise the `exception` parameter.
    """
    raise exception


def store_error_manager(call_trace: PipelineCallTraceStorage, exception: Exception) -> None:
    """Error manager for debugging.

    Args:
        call_trace (PipelineCallTraceStorage): Pipeline call results storage.
        exception (Exception): Exception raised during the pipeline call.
    """
    call_trace.write_error(exception)
