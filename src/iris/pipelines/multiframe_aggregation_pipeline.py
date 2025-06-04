import os
import traceback
from typing import Any, Dict, List, Optional, Union

from pydantic import validator

from iris._version import __version__
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisTemplate
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import (
    build_aggregation_multiframe_orb_output,
    build_simple_multiframe_aggregation_output,
)
from iris.orchestration.pipeline_dataclasses import PipelineMetadata, PipelineNode
from iris.orchestration.validators import pipeline_config_duplicate_node_name_check
from iris.pipelines.base_pipeline import BasePipeline, load_yaml_config
from iris.utils.base64_encoding import base64_decode_str


class MultiframeAggregationPipeline(BasePipeline):
    """
    Pipeline for multiframe iris template aggregation.
    Inherits shared logic from BasePipeline and implements input/output specifics.
    """

    DEFAULT_PIPELINE_CFG_PATH = os.path.join(os.path.dirname(__file__), "confs", "multiframe_aggregation_pipeline.yaml")
    PACKAGE_VERSION = __version__

    ORB_ENVIRONMENT = Environment(
        pipeline_output_builder=build_aggregation_multiframe_orb_output,
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
        subconfig_key: str = "templates_aggregation",
    ) -> None:
        """
        Initialize MultiframeAggregationPipeline with config and environment.
        Args:
            config (Union[Dict[str, Any], Optional[str]]): Pipeline config dict or YAML string.
            env (Environment): Pipeline environment.
            subconfig_key (str): The key to extract from the config dict. If provided, the config will be loaded from the subconfig key. Empty string means no subconfig is provided and the entire config is loaded.
        """
        deserialized_config = self.load_config(config, subconfig_key)
        super().__init__(deserialized_config, env)

    def run(self, templates: List[IrisTemplate], *args: Any, **kwargs: Any) -> Any:
        pipeline_input = {"templates": templates}
        return super().run(pipeline_input, *args, **kwargs)

    def _handle_input(self, pipeline_input: Any, *args, **kwargs) -> None:
        """
        Write the list of IrisTemplate objects to the call trace.
        Args:
            pipeline_input (Any): List of IrisTemplate objects.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        """
        templates = pipeline_input["templates"]
        self.call_trace.write_input(templates)

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

    @classmethod
    def load_config(
        cls, config: Union[Dict[str, Any], Optional[str]], keyword: str = "templates_aggregation"
    ) -> Dict[str, Any]:
        """
        Load and deserialize the pipeline configuration (for multiframe aggregation).

        Args:
            config: Either
                • a dict already containing your pipeline sections, or
                • a YAML string (or None) that will be loaded from disk.
            keyword: If non‐empty, the top‐level key to extract from the final dict.

        Returns:
            The sub-dict at `keyword` (or the entire dict if `keyword==""`).

        Raises:
            ValueError: if `keyword` is non-empty and not found in the config.
        """
        # 1) Figure out the raw dictionary
        if isinstance(config, dict):
            raw = config
        else:
            # config is a YAML string or None: load from the default multiframe_pipeline.yaml
            if config is None or config == "":  # noqa
                config = cls.DEFAULT_PIPELINE_CFG_PATH
            raw = load_yaml_config(config)

        # 2) If they asked for the whole dict, just return it
        if not keyword:
            return raw

        # 3) Otherwise, extract the sub‐key or raise once
        try:
            return raw[keyword]
        except KeyError:
            raise ValueError(f"MultiframeAggregation requires '{keyword}' in the configuration.")

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
        pipeline = None
        try:
            decoded_config_str = base64_decode_str(config)
            pipeline = cls(config=decoded_config_str)
        except Exception as exception:
            error = {
                "error_type": type(exception).__name__,
                "message": str(exception),
                "traceback": "".join(traceback.format_tb(exception.__traceback__)),
            }
        return {"agent": pipeline, "error": error}
