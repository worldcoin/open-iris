import os
import traceback
from typing import Any, Dict, List, Optional, Union

from pydantic import validator

from iris._version import __version__
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IrisTemplate, IrisTemplateWithId
from iris.io.errors import DifferentImageIdsTemplatesListLenError
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import (
    build_aggregation_templates_orb_output,
    build_simple_templates_aggregation_output,
)
from iris.orchestration.pipeline_dataclasses import PipelineMetadata, PipelineNode
from iris.orchestration.validators import pipeline_config_duplicate_node_name_check
from iris.pipelines.base_pipeline import BasePipeline, load_yaml_config
from iris.utils.base64_encoding import base64_decode_str


class TemplatesAggregationPipeline(BasePipeline):
    """
    Pipeline for iris templates aggregation.
    Inherits shared logic from BasePipeline and implements input/output specifics.
    """

    DEFAULT_PIPELINE_CFG_PATH = os.path.join(os.path.dirname(__file__), "confs", "templates_aggregation_pipeline.yaml")
    PACKAGE_VERSION = __version__

    ORB_ENVIRONMENT = Environment(
        pipeline_output_builder=build_aggregation_templates_orb_output,
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
            pipeline_output_builder=build_simple_templates_aggregation_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        ),
        subconfig_key: Optional[str] = "templates_aggregation",
    ) -> None:
        """
        Initialize TemplatesAggregationPipeline with config and environment.
        Args:
            config (Union[Dict[str, Any], Optional[str]]): Pipeline config dict or YAML string.
            env (Environment): Pipeline environment.
            subconfig_key (str): The key to extract from the config dict. If provided, the config will be loaded from the subconfig key. Empty string means no subconfig is provided and the entire config is loaded.
        """
        deserialized_config = self.load_config(config, keyword=subconfig_key)
        super().__init__(deserialized_config, env)

    def run(
        self, templates: List[IrisTemplate], image_ids: Optional[List[str]] = None, *args: Any, **kwargs: Any
    ) -> Any:
        try:
            # Validate input consistency
            if image_ids is not None and len(image_ids) != len(templates):
                raise DifferentImageIdsTemplatesListLenError(
                    f"Number of image_ids ({len(image_ids)}) must match number of templates ({len(templates)})"
                )

            # Create IrisTemplateWithId
            templates_with_ids = []
            for i, template in enumerate(templates):
                image_id = image_ids[i] if image_ids else f"frame_{i}"
                templates_with_ids.append(IrisTemplateWithId.from_template(template, image_id))
        except Exception as e:
            self.env.error_manager(self.call_trace, e)
            return self._handle_output(*args, **kwargs)

        pipeline_input = {"templates_with_ids": templates_with_ids}
        return super().run(pipeline_input, *args, **kwargs)

    def _handle_input(self, pipeline_input: Any, *args, **kwargs) -> None:
        """
        Write the list of IrisTemplateWithId objects to the call trace.
        Args:
            pipeline_input (Any): Dictionary containing 'templates_with_ids' key with List[IrisTemplateWithId] value.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        """
        templates_with_ids = pipeline_input["templates_with_ids"]
        self.call_trace.write_input(templates_with_ids)

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
        cls, config: Union[Dict[str, Any], Optional[str]], keyword: Optional[str] = "templates_aggregation"
    ) -> Dict[str, Any]:
        """
        Load and deserialize the pipeline configuration (for templates aggregation).

        Args:
            config: Either
                • a dict already containing your pipeline sections, or
                • a YAML string (or None) that will be loaded from disk.
            keyword: If None or empty string, the entire dict is returned. Otherwise, extracts the sub-dict at this key.

        Returns:
            The sub-dict at `keyword` (or the entire dict if `keyword` is None or empty).

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
        if keyword is None or keyword == "":  # noqa
            return raw

        # 3) Otherwise, extract the sub‐key or raise once
        try:
            return raw[keyword]
        except KeyError:
            raise ValueError(f"TemplatesAggregation requires '{keyword}' in the configuration.")

    @classmethod
    def load_from_config(
        cls, config: str
    ) -> Dict[str, Union["TemplatesAggregationPipeline", Optional[Dict[str, Any]]]]:
        """
        Given an iris config string in base64, initialise a TemplatesAggregationPipeline with this config.
        Args:
            config (str): Base64-encoded config string.
        Returns:
            Dict[str, Union[TemplatesAggregationPipeline, Optional[Dict[str, Any]]]]: Initialised pipeline and error (if any).
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
