import os
import traceback
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np

from iris._version import __version__
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.dataclasses import IrisTemplate
from iris.io.errors import IRISPipelineError, TemplatesAggregationPipelineError
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import (
    build_multiframe_iris_pipeline_orb_output,
    build_simple_iris_pipeline_orb_output,
    build_simple_multiframe_iris_pipeline_output,
    build_simple_templates_aggregation_output,
)
from iris.pipelines.base_pipeline import load_yaml_config
from iris.pipelines.iris_pipeline import IRISPipeline
from iris.pipelines.templates_aggregation_pipeline import TemplatesAggregationPipeline
from iris.utils.base64_encoding import base64_decode_str


class MultiframeIrisPipeline:
    """
    Pipeline that combines IRISPipeline and TemplatesAggregationPipeline.
    Takes a list of images and eye-side as input, processes each image through IRISPipeline,
    then aggregates the resulting templates using TemplatesAggregationPipeline.

    Uses a unified configuration with two distinct parts:
    - iris_pipeline: Configuration for individual image processing
    - templates_aggregation: Configuration for template aggregation
    """

    DEFAULT_PIPELINE_CFG_PATH = os.path.join(os.path.dirname(__file__), "confs", "multiframe_iris_pipeline.yaml")
    PACKAGE_VERSION = __version__

    ORB_ENVIRONMENT = Environment(
        pipeline_output_builder=build_multiframe_iris_pipeline_orb_output,
        error_manager=store_error_manager,
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )

    def __init__(
        self,
        config: Union[Dict[str, Any], Optional[str]] = None,
        env: Environment = Environment(
            pipeline_output_builder=build_simple_multiframe_iris_pipeline_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        ),
    ) -> None:
        """
        Initialize MultiframeIrisPipeline with unified config and environment.
        Args:
            config (Union[Dict[str, Any], Optional[str]]): Unified pipeline config dict or YAML string.
            env (Environment): Pipeline environment.
        """
        self.env = env
        self.iris_pipeline_config, self.templates_aggregation_pipeline_config = self.load_config(config)
        self.iris_pipeline, self.templates_aggregation_pipeline = self._initialize_pipelines(
            self.iris_pipeline_config, self.templates_aggregation_pipeline_config
        )

        # Derive iris template shape from the configuration
        self.iris_template_shape = self.derive_iris_template_shape_from_config(self.iris_pipeline_config)

        # Initialize call trace for the combined pipeline
        self.call_trace = self.env.call_trace_initialiser(nodes={}, pipeline_nodes=[])

    def estimate(
        self, imgs_data: List[np.ndarray], eye_side: Literal["left", "right"], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Wrap the `run` method to match the Orb system AI models call interface.

        Args:
            imgs_data (List[np.ndarray]): List of input images.
            eye_side (Literal["left", "right"]): Eye side for all images.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        Returns:
            Any: Output created by builder specified in environment.pipeline_output_builder.
        """
        return self.run(imgs_data, eye_side, *args, **kwargs)

    def run(self, imgs_data: List[np.ndarray], eye_side: Literal["left", "right"], *args: Any, **kwargs: Any) -> Any:
        """
        Process multiple images through the combined pipeline.
        Args:
            images (List[np.ndarray]): List of input images.
            eye_side (Literal["left", "right"]): Eye side for all images.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        Returns:
            Any: Output created by builder specified in environment.pipeline_output_builder.
        """
        self.call_trace.clean()

        pipeline_input = {"imgs_data": imgs_data, "eye_side": eye_side}
        self._handle_input(pipeline_input, *args, **kwargs)

        # Process individual images through iris pipeline
        try:
            iris_templates, _ = self._run_iris_pipeline(imgs_data, eye_side)

            # Run aggregation pipeline
            _ = self._run_aggregation_pipeline(iris_templates)

        except Exception as e:
            self._handle_pipeline_error(e)

        # return the aggregation pipeline output
        return self._handle_output(*args, **kwargs)

    @classmethod
    def load_config(cls, config: Union[Dict[str, Any], Optional[str]]) -> Dict[str, Any]:
        """
        Load and deserialize the pipeline configuration (for templates aggregation).

        Args:
            config: Either
                • a dict already containing your pipeline sections, or
                • a YAML string (or None) that will be loaded from disk.

        Returns:
            Dict[str, Any]: Dictionary containing the iris_pipeline and templates_aggregation_pipeline configurations.
        """
        # 1) Figure out the raw dictionary
        if isinstance(config, dict):
            raw = config
        else:
            # config is a YAML string or None: load from the default multiframe_pipeline.yaml
            if config is None or config == "":  # noqa
                config = cls.DEFAULT_PIPELINE_CFG_PATH
            raw = load_yaml_config(config)

        # 2) Split the config into iris_pipeline and templates_aggregation_pipeline
        for key in ["iris_pipeline", "templates_aggregation_pipeline"]:
            if key not in raw:
                raise ValueError(f"MultiframeIrisPipeline requires '{key}' in the configuration.")

        return raw["iris_pipeline"], raw["templates_aggregation_pipeline"]

    @classmethod
    def load_from_config(cls, config: str) -> Dict[str, Union["MultiframeIrisPipeline", Optional[Dict[str, Any]]]]:
        """
        Given an iris config string in base64, initialise a MultiframeIrisPipeline with this config.
        Args:
            config (str): Base64-encoded config string.
        Returns:
            Dict[str, Union[MultiframeIrisPipeline, Optional[Dict[str, Any]]]]: Initialised pipeline and error (if any).
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

    @staticmethod
    def derive_iris_template_shape_from_config(iris_pipeline_config: Dict[str, Any]) -> Tuple[int, int, int, int]:
        """
        Derive the iris template shape from the pipeline configuration.

        The shape is determined by scanning the filter_bank node configuration:
        - n_rows and n_cols from probe_schemas (first two dimensions)
        - number of filters (third dimension)
        - number of probe_schemas (fourth dimension)

        Returns:
            Tuple[int, int, int, int]: The iris template shape (n_rows, n_cols, n_filters, n_probe_schemas)
        """
        # Find the filter_bank node in the iris pipeline configuration
        pipeline_nodes = iris_pipeline_config.get("pipeline", [])
        filter_bank_node = None

        for node in pipeline_nodes:
            if node.get("name") == "filter_bank":
                filter_bank_node = node
                break

        if filter_bank_node is None:
            raise ValueError("filter_bank node not found in iris pipeline configuration")

        # Extract probe schema parameters
        algorithm_params = filter_bank_node.get("algorithm", {}).get("params", {})
        probe_schemas = algorithm_params.get("probe_schemas", [])

        if not probe_schemas:
            raise ValueError("No probe_schemas found in filter_bank configuration")

        n_probe_schemas = len(probe_schemas)
        if n_probe_schemas == 0:
            raise ValueError("No probe_schemas found in filter_bank configuration")

        # Get n_rows and n_cols from the first probe schema
        first_probe_schema = probe_schemas[0]
        probe_params = first_probe_schema.get("params", {})
        n_rows = probe_params.get("n_rows")
        n_cols = probe_params.get("n_cols")

        if n_rows is None or n_cols is None:
            raise ValueError("n_rows or n_cols not found in probe schema configuration")

        # Count the number of filters
        filters = algorithm_params.get("filters", [])
        n_filters = len(filters)

        if n_filters == 0:
            raise ValueError("No filters found in filter_bank configuration")

        return (n_rows, n_cols, n_filters, n_probe_schemas)

    def _run_iris_pipeline(
        self, imgs_data: List[np.ndarray], eye_side: Literal["left", "right"]
    ) -> Tuple[List[IrisTemplate], List[Any]]:
        """
        Process multiple images through the iris pipeline.

        Args:
            imgs_data (List[np.ndarray]): List of input images.
            eye_side (Literal["left", "right"]): Eye side for all images.

        Returns:
            Tuple[List[IrisTemplate], List[Any]]: Tuple containing:
                - List of iris templates extracted from each image
                - List of individual pipeline outputs for each image
        """
        iris_templates = []
        individual_templates_output = []  # Collect individual template outputs

        for i, img in enumerate(imgs_data):
            iris_pipeline_output = self.iris_pipeline.run(img, eye_side)
            individual_templates_output.append(iris_pipeline_output)

            # if there was an error - re-raise it and let the caller handle it
            if iris_pipeline_output["error"] is not None:
                # store the error in the call_trace for this frame
                self.call_trace.write("individual_frames", individual_templates_output)
                # re-raise the error
                message = f"Error in IrisPipeline for frame {i}: see individual_frames for details"
                raise IRISPipelineError(message)

            template = iris_pipeline_output["iris_template"]
            if isinstance(template, dict):
                template = IrisTemplate.deserialize(template, self.iris_template_shape)
            elif template is None:
                pass  # TODO: handle this case
            else:
                # template is already a IrisTemplate object
                pass

            iris_templates.append(template)

        # Write individual frames to call_trace
        self.call_trace.write("individual_frames", individual_templates_output)

        return iris_templates, individual_templates_output

    def _run_aggregation_pipeline(self, iris_templates: List[IrisTemplate]) -> Any:
        """
        Run the aggregation pipeline on a list of iris templates.

        Args:
            iris_templates (List[IrisTemplate]): List of iris templates to aggregate.

        Returns:
            Any: Output from the aggregation pipeline.
        """
        aggregation_pipeline_output = self.templates_aggregation_pipeline.run(iris_templates)

        # Store aggregation result in call_trace
        self.call_trace.write("aggregation_result", aggregation_pipeline_output)

        if aggregation_pipeline_output["error"] is not None:
            message = "Error in TemplatesAggregationPipeline: see aggregation_result for details"
            raise TemplatesAggregationPipelineError(message)

        return aggregation_pipeline_output

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

    def _handle_input(self, pipeline_input: Any, *args, **kwargs) -> None:
        """
        Write the list of IrisTemplate objects to the call trace.
        Args:
            pipeline_input (Any): List of IrisTemplate objects.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        """
        img_data = pipeline_input["imgs_data"]
        eye_side = pipeline_input["eye_side"]

        # Check that img_data is a list of numpy arrays
        if not isinstance(img_data, (list, tuple)):
            raise ValueError("imgs_data must be a list of numpy arrays.")
        if not all(isinstance(img, np.ndarray) for img in img_data):
            raise ValueError("imgs_data must be a list of numpy arrays.")

        # Check that eye_side is 'left' or 'right'
        if eye_side not in ("left", "right"):
            raise ValueError("eye_side must be either 'left' or 'right'.")
        self.call_trace.write_input(pipeline_input)

    def _handle_pipeline_error(self, error: Exception, allow_skip: bool = False) -> bool:
        """
        Default error handling for pipeline execution. Can be overridden by subclasses.
        Args:
            error (Exception): The exception raised during pipeline execution.
            allow_skip (bool): Whether to allow skipping the error.
        Returns:
            bool: True if the error should be skipped, False otherwise.
        """
        self.env.error_manager(self.call_trace, error)
        return allow_skip

    @staticmethod
    def _initialize_pipelines(
        iris_pipeline_config: Dict[str, Any], templates_aggregation_pipeline_config: Dict[str, Any]
    ) -> Tuple[IRISPipeline, TemplatesAggregationPipeline]:
        """
        Initialize the iris and templates aggregation pipelines.
        Args:
            iris_pipeline_config (Dict[str, Any]): The configuration for the iris pipeline.
            templates_aggregation_pipeline_config (Dict[str, Any]): The configuration for the templates aggregation pipeline.
        Returns:
            Tuple[IRISPipeline, TemplatesAggregationPipeline]: The initialized iris and templates aggregation pipelines.
        """
        # Initialize sub-pipelines with their respective configurations
        # We use a "simple" env that does not serialize the iris template within the individual IrisPipeline
        # Serialization will then be controlled by the MultiframeIrisPipeline env
        iris_pipeline = IRISPipeline(
            config=iris_pipeline_config,
            env=Environment(
                pipeline_output_builder=build_simple_iris_pipeline_orb_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
        )

        templates_aggregation_pipeline = TemplatesAggregationPipeline(
            config=templates_aggregation_pipeline_config,
            env=Environment(
                pipeline_output_builder=build_simple_templates_aggregation_output,
                error_manager=store_error_manager,
                call_trace_initialiser=PipelineCallTraceStorage.initialise,
            ),
            subconfig_key=None,
        )

        return iris_pipeline, templates_aggregation_pipeline
