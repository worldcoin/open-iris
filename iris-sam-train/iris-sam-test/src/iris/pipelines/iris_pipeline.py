from __future__ import annotations

import os
import pydoc
import traceback
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import yaml
from pydantic import validator

import iris  # noqa: F401
import iris.nodes.validators.cross_object_validators
import iris.nodes.validators.object_validators
from iris.callbacks.pipeline_trace import NodeResultsWriter, PipelineCallTraceStorage, PipelineCallTraceStorageError
from iris.io.class_configs import Algorithm
from iris.io.dataclasses import IRImage
from iris.io.errors import IRISPipelineError
from iris.orchestration.environment import Environment
from iris.orchestration.error_managers import store_error_manager
from iris.orchestration.output_builders import build_orb_output, build_simple_debugging_output, build_simple_orb_output
from iris.orchestration.pipeline_dataclasses import PipelineClass, PipelineMetadata, PipelineNode
from iris.orchestration.validators import pipeline_config_duplicate_node_name_check
from iris.utils.base64_encoding import base64_decode_str


class IRISPipeline(Algorithm):
    """Implementation of a fully configurable iris recognition pipeline."""

    DEBUGGING_ENVIRONMENT = Environment(
        pipeline_output_builder=build_simple_debugging_output,
        error_manager=store_error_manager,
        disabled_qa=[
            iris.nodes.validators.object_validators.Pupil2IrisPropertyValidator,
            iris.nodes.validators.object_validators.OffgazeValidator,
            iris.nodes.validators.object_validators.OcclusionValidator,
            iris.nodes.validators.object_validators.IsPupilInsideIrisValidator,
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
        """IRISPipeline parameters, all derived from the input `config`."""

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
            pipeline_output_builder=build_simple_orb_output,
            error_manager=store_error_manager,
            call_trace_initialiser=PipelineCallTraceStorage.initialise,
        ),
    ) -> None:
        """Initialise IRISPipeline.

        Args:
            config (Union[Dict[str, Any], Optional[str]]): Input configuration, as a YAML-formatted string or dictionary specifying all nodes configuration. Defaults to None, which loads the default config.
            env (Environment, optional): Environment properties. Defaults to Environment(pipeline_output_builder=build_simple_output, error_manager=store_error_manager, call_trace_initialiser=PipelineCallTraceStorage).
        """
        deserialized_config = self.load_config(config) if isinstance(config, str) or config is None else config
        super().__init__(**deserialized_config)
        self._check_pipeline_coherency()

        self.env = env
        self.nodes = self.instanciate_nodes()
        self.call_trace = self.env.call_trace_initialiser(nodes=self.nodes, pipeline_nodes=self.params.pipeline)

    def estimate(self, img_data: np.ndarray, eye_side: Literal["left", "right"]) -> Any:
        """Wrap the `run` method to match the Orb system AI models call interface.

        Args:
            img_data (np.ndarray): Image data.
            eye_side (Literal["left", "right"]): Eye side.

        Returns:
            Any: Output created by builder specified in environment.pipeline_output_builder.
        """
        return self.run(img_data=img_data, eye_side=eye_side)

    def run(self, img_data: np.ndarray, eye_side: Literal["left", "right"], bboxes:np.ndarray = np.empty(0)) -> Any:
        """Generate template.

        Args:
            img_data (np.ndarray): Infrared image as a numpy array.
            eye_side (Literal["left", "right"]): Eye side.

        Returns:
            Any: Output created by builder specified in environment.pipeline_output_builder.
        """
        self.call_trace.clean()

        ir_image = IRImage(img_data=img_data, eye_side=eye_side, bboxes=bboxes)
        self.call_trace.write_input(ir_image)

        for node in self.params.pipeline:
            input_kwargs = {}
            for node_input in node.inputs:
                if isinstance(node_input.source_node, list):
                    input_kwargs[node_input.name] = []
                    for src_node in node_input.source_node:
                        if src_node.index is not None:
                            input_kwargs[node_input.name].append(self.call_trace[src_node.name][src_node.index])
                        else:
                            input_kwargs[node_input.name].append(self.call_trace[src_node.name])

                else:
                    input_kwargs[node_input.name] = self.call_trace[node_input.source_node]
                    if node_input.index is not None:
                        input_kwargs[node_input.name] = input_kwargs[node_input.name][node_input.index]
            try:
                if self.call_trace[node.name] is not None:
                    continue
                _ = self.nodes[node.name](**input_kwargs)

            except (PipelineCallTraceStorageError, KeyError):
                if pydoc.locate(node.algorithm.class_name) in self.env.disabled_qa:
                    continue
                self.env.error_manager(self.call_trace, ValueError(f"Could not find node {node.name}."))
                break

            except Exception as e:
                self.env.error_manager(self.call_trace, e)
                break

        return self.env.pipeline_output_builder(self.call_trace)

    def _init_pipeline_tracing(self) -> PipelineCallTraceStorage:
        """Instantiate mechanisms for intermediate results tracing.

        Returns:
            PipelineCallTraceStorage: Pipeline intermediate and final results storage.
        """
        call_trace = self.env.call_trace_class(results_names=self.nodes.keys())

        for algorithm_name, algorithm_object in self.nodes.items():
            algorithm_object._callbacks.append(NodeResultsWriter(call_trace, algorithm_name))

        return call_trace

    def instanciate_nodes(self) -> Dict[str, Algorithm]:
        """Given a list of PipelineNode, return the associated instanciated nodes.

        NOTE: All nodes of type listed in self.env.disabled_qa will be filtered out. This allows one config file to be used in various QA standards levels.

        Returns:
            Dict[str, Algorithm]: instanciated nodes.
        """
        instanciated_pipeline = self.instanciate_pipeline()
        nodes = {
            node.name: self.instanciate_node(
                node_class=node.algorithm.class_name,
                algorithm_params=node.algorithm.params,
                callbacks=node.callbacks,
            )
            for node in instanciated_pipeline
        }
        nodes = {node_name: node for node_name, node in nodes.items() if type(node) not in self.env.disabled_qa}
        return nodes

    def instanciate_pipeline(self) -> List[PipelineNode]:
        """Given a list of PipelineNodes, crawl the parameters and instanciate the PipelineClass available.

        Returns:
            List[PipelineNode]: pipeline with instanciated parameters
        """
        instanciated_pipeline = []
        for node in self.params.pipeline:
            current_node = node
            for param_name, param_value in node.algorithm.params.items():
                if isinstance(param_value, (tuple, list)):
                    for i, value in enumerate(param_value):
                        if isinstance(value, PipelineClass):
                            current_node.algorithm.params[param_name][i] = self.instanciate_class(
                                class_name=value.class_name, kwargs=value.params
                            )
                elif isinstance(param_value, PipelineClass):
                    current_node.algorithm.params[param_name] = self.instanciate_class(
                        class_name=param_value.class_name, kwargs=param_value.params
                    )
            instanciated_pipeline.append(current_node)
        return instanciated_pipeline

    def instanciate_node(
        self, node_class: str, algorithm_params: Dict[str, Any], callbacks: Optional[List[PipelineClass]]
    ) -> Algorithm:
        """Instanciate an Algorithm from its class, kwargs and optional Callbacks.

        NOTE: All callbacks of type listed in self.env.disabled_qa will be filtered out. This allows one config file to be used in various QA standards levels.

        Args:
            node_class (str): Node's class.
            algorithm_params (Dict[str, Any]): Node's kwargs.
            callbacks (Optional[List[PipelineClass]]): list of callbacks.

        Returns:
            Algorithm: instanciated node.
        """
        if callbacks is not None and len(callbacks):
            instanciated_callbacks = [self.instanciate_class(cb.class_name, cb.params) for cb in callbacks]
            instanciated_callbacks = [cb for cb in instanciated_callbacks if type(cb) not in self.env.disabled_qa]

            algorithm_params = {**algorithm_params, **{"callbacks": instanciated_callbacks}}

        return self.instanciate_class(node_class, algorithm_params)

    def instanciate_class(self, class_name: str, kwargs: Dict[str, Any]) -> Callable:
        """Instanciate a class from its string definition and its kwargs.

        This function relies on pydoc.locate, a safe way to instanciate a class from its string definition, which itself relies on pydoc.safe_import.

        Args:
            class_name (str): name of the class.
            kwargs (Dict): kwargs to pass to the class at instanciation time

        Returns:
            Callable: the instanciated class

        Raises:
            IRISPipelineError: Raised if the class cannot be located.
        """
        object_class = pydoc.locate(class_name)

        if object_class is None:
            raise IRISPipelineError(f"Could not locate class {class_name}")

        return object_class(**kwargs)

    def _check_pipeline_coherency(self) -> None:
        """Check the pipeline configuration coherency.

        Raises:
            IRISPipelineError: Raised if a node's inputs are not declared beforehands
        """
        parent_names = [PipelineCallTraceStorage.INPUT_KEY_NAME]
        for node in self.params.pipeline:
            for input_node in node.inputs:
                if isinstance(input_node.source_node, (tuple, list)):
                    for input_element in input_node.source_node:
                        if input_element.name not in parent_names:
                            raise IRISPipelineError(
                                f"Pipeline configuration incoherent. Node {node.name} has input "
                                f"{input_element.name} not declared prior. Please fix IRISPipeline configuration."
                            )
                elif input_node.source_node not in parent_names:
                    raise IRISPipelineError(
                        f"Pipeline configuration incoherent. Node {node.name} has input "
                        f"{input_node.source_node} not declared prior. Please fix IRISPipeline configuration."
                    )

            parent_names.append(node.name)

    @staticmethod
    def load_config(config: Optional[str]) -> Dict[str, Any]:
        """Convert the input configuration string into a dictionary for deserialisation. If no config is given, load the default config.

        Args:
            config (Optional[str]): YAML-formatted input configuration string.

        Raises:
            IRISPipelineError: Raised if the input config is not a string, or is not correctly YAML-formatted.

        Returns:
            Dict[str, Any]: Configuration as a dictionary.
        """
        if config is None or config == "":
            with open(os.path.join(os.path.dirname(__file__), "confs", "pipeline.yaml"), "r") as f:
                deserialized_config = yaml.safe_load(f)
        elif isinstance(config, str):
            try:
                deserialized_config = yaml.safe_load(config)
            except yaml.parser.ParserError:
                raise IRISPipelineError(
                    "IRISPipeline requires a YAML-formatted configuration string. Please check the format"
                )
        else:
            raise IRISPipelineError(
                "IRISPipeline requires a YAML-formatted configuration string. Please check the type"
            )

        return deserialized_config

    @classmethod
    def load_from_config(cls, config: str) -> Dict[str, Union[IRISPipeline, Optional[Dict[str, Any]]]]:
        """Given an iris config string in base64, initialise an IRISPipeline with config this config.

        Args:
            config (str): an iris str configs in base64

        Returns:
            Dict[str, Union[IRISPipeline, Optional[Dict[str, Any]]]]: Initialised iris pipeline and standard error output.
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
