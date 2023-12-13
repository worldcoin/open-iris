import pytest
from pydantic import ValidationError

from iris.orchestration.environment import Environment


def test_environment_object_creation() -> None:
    def build_func(trace):
        return trace["doesntmatter"]

    def err_man_func(trace, exception):
        pass

    def call_trace_init_func(nodes, pipelines):
        pass

    _ = Environment(
        pipeline_output_builder=build_func, error_manager=err_man_func, call_trace_initialiser=call_trace_init_func
    )


def test_environment_raises_an_error_when_build_function_not_provided() -> None:
    with pytest.raises(ValidationError):
        _ = Environment(pipeline_output_builder=None)
