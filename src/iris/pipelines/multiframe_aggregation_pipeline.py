import os
import traceback
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import validator

import iris  # noqa: F401
import iris.nodes.validators.cross_object_validators
import iris.nodes.validators.object_validators
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.class_configs import Algorithm
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import build_orb_output, build_simple_multiframe_aggregation_output
from iris.orchestration.pipeline_dataclasses import PipelineMetadata, PipelineNode
from iris.orchestration.validators import pipeline_config_duplicate_node_name_check
from iris.pipelines.base_pipeline import BasePipeline
from iris.utils.base64_encoding import base64_decode_str


def _load_yaml_config(config: Optional[str], default_path: str) -> Dict[str, Any]:
    if config is None or not config:
        with open(default_path, "r") as f:
            return yaml.safe_load(f)
    elif isinstance(config, str):
        try:
            return yaml.safe_load(config)
        except yaml.YAMLError:
            raise ValueError("Requires a YAML-formatted configuration string. Please check the format")
    else:
        raise ValueError("Requires a YAML-formatted configuration string. Please check the type")


class MultiframeAggregationPipeline(BasePipeline):
    """
    Pipeline for multiframe iris template aggregation.
    Inherits shared logic from BasePipeline and implements input/output specifics.
    """

    DEBUGGING_ENVIRONMENT = Environment(
        pipeline_output_builder=build_simple_multiframe_aggregation_output,
        error_manager=store_error_manager,
        disabled_qa=[
            iris.nodes.validators.object_validators.Pupil2IrisPropertyValidator,
            iris.nodes.validators.object_validators.OffgazeValidator,
            iris.nodes.validators.object_validators.OcclusionValidator,
            iris.nodes.validators.object_validators.IsPupilInsideIrisValidator,
            iris.nodes.validators.object_validators.SharpnessValidator,
            iris.nodes.validators.object_validators.IsMaskTooSmallValidator,
            iris.nodes.validators.cross_object_validators.EyeCentersInsideImageValidator,
            iris.nodes.validators.cross_object_validators.ExtrapolatedPolygonsInsideImageValidator,
        ],
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )

    ORB_ENVIRONMENT = Environment(
        pipeline_output_builder=build_orb_output,
        error_manager=store_error_manager,
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )

    class Parameters(Algorithm.Parameters):
        metadata: PipelineMetadata
        pipeline: List[PipelineNode]
        _config_duplicate_node_name_check = validator("pipeline", allow_reuse=True)(
            pipeline_config_duplicate_node_name_check
        )

    __parameters_type__ = Parameters

    def __init__(
        self,
        config: Union[Dict[str, Any], Optional[str]] = None,
        env: Environment = Environment(
            pipeline_output_builder=build_simple_multiframe_aggregation_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        ),
    ) -> None:
        """
        Initialize MultiframeAggregationPipeline with config and environment.
        Args:
            config (Union[Dict[str, Any], Optional[str]]): Pipeline config dict or YAML string.
            env (Environment): Pipeline environment.
        """
        deserialized_config = self.load_config(config) if isinstance(config, str) or config is None else config
        super().__init__(deserialized_config, env)

    def _handle_input(self, pipeline_input: Any, *args, **kwargs) -> None:
        """
        Write the list of IrisTemplate objects to the call trace.
        Args:
            pipeline_input (Any): List of IrisTemplate objects.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        """
        self.call_trace.write_input(pipeline_input)

    def _handle_output(self, *args, **kwargs) -> Any:
        """
        Build and return the pipeline output using the environment's output builder.
        Args:
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        Returns:
            Any: Output as built by the pipeline_output_builder.
        """
        return self.env.pipeline_output_builder(self.call_trace)

    @staticmethod
    def load_config(config: Optional[str], keyword: str = "templates_aggregation") -> Dict[str, Any]:
        """
        Load and deserialize the pipeline configuration for multiframe aggregation.
        Args:
            config (Optional[str]): YAML string or None for default config.
            keyword (str): The key to extract from the config dict.
        Returns:
            Dict[str, Any]: Deserialized config dict for the aggregation pipeline.
        """
        deserialized_config = _load_yaml_config(
            config, os.path.join(os.path.dirname(__file__), "confs", "multiframe_pipeline.yaml")
        )
        if keyword not in deserialized_config:
            raise ValueError(
                f"MultiframeAggregation requires a valid keyword in the configuration file. Please check the keyword: {keyword}"
            )
        aggregation_config = deserialized_config[keyword]
        return aggregation_config

    @classmethod
    def load_from_config(
        cls, config: str
    ) -> Dict[str, Union["MultiframeAggregationPipeline", Optional[Dict[str, Any]]]]:
        """
        Given an iris config string in base64, initialise a MultiframeAggregationPipeline with this config.
        Args:
            config (str): Base64-encoded config string.
        Returns:
            Dict[str, Union[MultiframeAggregationPipeline, Optional[Dict[str, Any]]]]: Initialised pipeline and error (if any).
        """
        error = None
        iris_pipeline = None
        try:
            decoded_config_str = base64_decode_str(config)
            iris_pipeline = cls(config=decoded_config_str)
        except Exception as exception:
            error = {
                "error_type": type(exception).__name__,
                "message": str(exception),
                "traceback": "".join(traceback.format_tb(exception.__traceback__)),
            }
        return {"agent": iris_pipeline, "error": error}
