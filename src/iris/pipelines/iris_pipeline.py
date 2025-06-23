from __future__ import annotations

import os
import pydoc
import re
import traceback
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import validator

import iris  # noqa: F401
import iris.nodes.validators.cross_object_validators
import iris.nodes.validators.object_validators
from iris._version import __version__
from iris.callbacks.pipeline_trace import PipelineCallTraceStorage, PipelineCallTraceStorageError
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IRImage
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import (
    build_iris_pipeline_orb_output,
    build_simple_iris_pipeline_debugging_output,
    build_simple_iris_pipeline_orb_output,
)
from iris.orchestration.pipeline_dataclasses import PipelineMetadata, PipelineNode
from iris.orchestration.validators import pipeline_config_duplicate_node_name_check, pipeline_metadata_version_check
from iris.pipelines.base_pipeline import BasePipeline, load_yaml_config
from iris.utils.base64_encoding import base64_decode_str


class IRISPipeline(BasePipeline):
    """
    Implementation of a fully configurable iris recognition pipeline.
    Inherits shared logic from BasePipeline and implements input/output specifics.
    """

    DEFAULT_PIPELINE_CFG_PATH = os.path.join(os.path.dirname(__file__), "confs", "pipeline.yaml")
    PACKAGE_VERSION = __version__

    DEBUGGING_ENVIRONMENT = Environment(
        pipeline_output_builder=build_simple_iris_pipeline_debugging_output,
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
        pipeline_output_builder=build_iris_pipeline_orb_output,
        error_manager=store_error_manager,
        call_trace_initialiser=PipelineCallTraceStorage.initialise,
    )

    class Parameters(Algorithm.Parameters):
        """IRISPipeline parameters, all derived from the input `config`."""

        metadata: PipelineMetadata
        pipeline: List[PipelineNode]

        _config_duplicate_node_name_check = validator("pipeline", allow_reuse=True)(
            pipeline_config_duplicate_node_name_check
        )

        @validator("metadata", allow_reuse=True)
        def _version_check(cls, v, values, **kwargs):
            # support dynamic subclassing
            pipeline_cls = getattr(cls, "__outer_class__", IRISPipeline)  # assigned in BasePipeline.__init_subclass__
            pipeline_metadata_version_check(cls, v.iris_version, values, expected_version=pipeline_cls.PACKAGE_VERSION)
            return v

    __parameters_type__ = Parameters

    def __init__(
        self,
        config: Union[Dict[str, Any], Optional[str]] = None,
        env: Environment = Environment(
            pipeline_output_builder=build_simple_iris_pipeline_orb_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        ),
    ) -> None:
        """
        Initialize IRISPipeline with config and environment.
        Args:
            config (Union[Dict[str, Any], Optional[str]]): Pipeline config dict or YAML string.
            env (Environment): Pipeline environment.
        """
        deserialized_config = self.load_config(config) if isinstance(config, str) or config is None else config
        super().__init__(deserialized_config, env)

    def estimate(self, img_data: np.ndarray, eye_side: Literal["left", "right"], *args, **kwargs) -> Any:
        """Wrap the `run` method to match the Orb system AI models call interface.

        Args:
            img_data (np.ndarray): Input image data.
            eye_side (Literal["left", "right"]): Eye side.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.

        Returns:
            Any: Output created by builder specified in environment.pipeline_output_builder.
        """
        return self.run(img_data, eye_side, *args, **kwargs)

    def run(self, img_data: np.ndarray, eye_side: Literal["left", "right"], *args, **kwargs) -> Any:
        """
        Wrap the `run` method to match the Orb system AI models call interface.

        Args:
            img_data (np.ndarray): Input Infrared image data.
            eye_side (Literal["left", "right"]): Eye side.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.

        Returns:
            Any: Output created by builder specified in environment.pipeline_output_builder.
        """
        pipeline_input = {"img_data": img_data, "eye_side": eye_side}
        return super().run(pipeline_input, *args, **kwargs)

    def _handle_input(self, pipeline_input: Any, *args, **kwargs) -> None:
        """
        Write the IR image input to the call trace.
        Args:
            pipeline_input (dict): Should contain 'img_data' and 'eye_side'.
            *args: Optional positional arguments for extensibility.
            **kwargs: Optional keyword arguments for extensibility.
        """
        ir_image = IRImage(img_data=pipeline_input["img_data"], eye_side=pipeline_input["eye_side"])
        self.call_trace.write_input(ir_image)

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

    def _handle_node_error(self, node: PipelineNode, error: Exception) -> bool:
        """
        Custom error handling for node execution in IRISPipeline.
        Handles PipelineCallTraceStorageError and KeyError by checking if the node's class is in disabled_qa.
        Args:
            node (PipelineNode): The node where the error occurred.
            error (Exception): The exception raised during node execution.
        Returns:
            bool: True if the error was skipped, False otherwise.
        """
        # If the error is a PipelineCallTraceStorageError or KeyError and the node is in disabled_qa, skip
        if isinstance(error, (PipelineCallTraceStorageError, KeyError)):
            node_class = pydoc.locate(node.algorithm.class_name)
            if node_class in getattr(self.env, "disabled_qa", []):
                return True
            # Otherwise, treat as a missing node error
            self.env.error_manager(self.call_trace, ValueError(f"Could not find node {node.name}."))
            return False
        else:
            # For all other errors, use the default error manager
            self.env.error_manager(self.call_trace, error)
            return False

    @classmethod
    def load_config(cls, config: Optional[str]) -> Dict[str, Any]:
        """
        Load and deserialize the pipeline configuration.
        Args:
            config (Optional[str]): YAML string or None for default config.
        Returns:
            Dict[str, Any]: Deserialized config dict.
        """
        if config is None or config == "":  # noqa
            config = cls.DEFAULT_PIPELINE_CFG_PATH
        deserialized_config = load_yaml_config(config)
        return deserialized_config

    @classmethod
    def load_from_config(cls, config: str) -> Dict[str, Union["IRISPipeline", Optional[Dict[str, Any]]]]:
        """
        Given an iris config string in base64, initialise an IRISPipeline with this config.
        Args:
            config (str): Base64-encoded config string.
        Returns:
            Dict[str, Union[IRISPipeline, Optional[Dict[str, Any]]]]: Initialised pipeline and error (if any).
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

    def update_config(self, config: str) -> None:
        """Update the pipeline configuration based on the provided base64-encoded string.

        Args:
            config (str): Base64-encoded string of the new configuration.
        """
        if not re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", config):
            raise ValueError("Invalid base64-encoded string")

        decoded_config_str = base64_decode_str(config)
        config_dict = self.load_config(decoded_config_str)

        params = self.__parameters_type__(**config_dict)
        self._check_pipeline_coherency(params)
        self.params = params

        self.nodes = self._instanciate_nodes()
        self.call_trace = self.env.call_trace_initialiser(nodes=self.nodes, pipeline_nodes=self.params.pipeline)
