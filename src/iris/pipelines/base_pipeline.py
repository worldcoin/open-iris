import abc
import os
from typing import Any, Dict, Generic, List, Optional, TypeVar

import yaml

from iris.callbacks.pipeline_trace import PipelineCallTraceStorage
from iris.io.class_configs import Algorithm, instantiate_class_from_name
from iris.orchestration.environment import Environment
from iris.orchestration.pipeline_dataclasses import PipelineNode

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


def load_yaml_config(config: Optional[str]) -> Dict[str, Any]:
    """
    Loads YAML config from a YAML string or a path to a YAML file.

    Args:
        config: YAML string or path to YAML file.

    Returns:
        Dict[str, Any]: Deserialized config dict.

    Raises:
        ValueError: If config is None, not a string, or can't be parsed.
    """
    if config is None:
        raise ValueError("No configuration provided.")

    if not isinstance(config, str):
        raise ValueError("Config must be a YAML string or a path to a YAML file.")

    # Check if it's a path to a file
    if os.path.isfile(config):
        with open(config, "r") as f:
            return yaml.safe_load(f)
    else:
        try:
            return yaml.safe_load(config)
        except (yaml.YAMLError, yaml.parser.ParserError):
            raise ValueError("Provided string is not valid YAML or a valid path to a YAML file.")


class BasePipeline(Algorithm, Generic[InputType, OutputType], abc.ABC):
    """
    Generic base class for IRIS pipelines, abstracting shared logic for pipeline execution,
    node instantiation, environment setup, and call trace management.
    Subclasses should implement input/output specifics via _handle_input and _handle_output.
    """

    PACKAGE_VERSION: str

    def __init__(
        self,
        config: Dict[str, Any],
        env: Environment,
    ) -> None:
        """
        Initialize the pipeline with configuration and environment.
        Args:
            config (Dict[str, Any]): Pipeline configuration dictionary.
            env (Environment): Pipeline environment (output builder, error manager, etc.).
        """
        super().__init__(**config)
        self.env = env
        self._check_pipeline_coherency(self.params)
        self.nodes = self._instanciate_nodes()
        self.call_trace = self.env.call_trace_initialiser(nodes=self.nodes, pipeline_nodes=self.params.pipeline)

    def __init_subclass__(cls) -> None:
        super().__init_subclass__()
        if not hasattr(cls, "PACKAGE_VERSION") or not isinstance(cls.PACKAGE_VERSION, str) or not cls.PACKAGE_VERSION:
            raise TypeError(f"{cls.__name__} must define a non-empty string PACKAGE_VERSION class attribute.")
        if hasattr(cls, "Parameters"):
            cls.Parameters.__outer_class__ = cls

    def estimate(self, pipeline_input: InputType, *args: Any, **kwargs: Any) -> OutputType:
        """Wrap the `run` method to match the Orb system AI models call interface.

        Args:
            pipeline_input (InputType): Input to the pipeline (type defined by subclass).
            *args (Any): Optional positional arguments for extensibility.
            **kwargs (Any): Optional keyword arguments for extensibility.
        Returns:
            OutputType: Output from the pipeline (type defined by subclass).
        """
        return self.run(pipeline_input, *args, **kwargs)

    def run(self, pipeline_input: InputType, *args: Any, **kwargs: Any) -> OutputType:
        """
        Main pipeline execution loop. Handles input, node execution, and output.
        Args:
            pipeline_input (InputType): Input to the pipeline (type defined by subclass).
            *args (Any): Optional positional arguments for extensibility.
            **kwargs (Any): Optional keyword arguments for extensibility.
        Returns:
            OutputType: Output from the pipeline (type defined by subclass).
        """
        self.call_trace.clean()
        self._handle_input(pipeline_input, *args, **kwargs)
        for node in self.params.pipeline:
            input_kwargs = self._get_node_inputs(node)
            try:
                if self.call_trace[node.name] is not None:
                    continue
                _ = self.nodes[node.name](**input_kwargs)
            except Exception as e:
                skip_error = self._handle_node_error(node, e)
                if skip_error:
                    continue
                break
        return self._handle_output(*args, **kwargs)

    @abc.abstractmethod
    def _handle_input(self, pipeline_input: InputType, *args: Any, **kwargs: Any) -> None:
        """
        Write the pipeline input to the call trace. To be implemented by subclasses.
        Args:
            pipeline_input (InputType): Input to the pipeline.
            *args (Any): Optional positional arguments for extensibility.
            **kwargs (Any): Optional keyword arguments for extensibility.
        """
        pass

    @abc.abstractmethod
    def _handle_output(self, *args, **kwargs) -> OutputType:
        """
        Build and return the pipeline output from the call trace. To be implemented by subclasses.
        Args:
            *args (Any): Optional positional arguments for extensibility.
            **kwargs (Any): Optional keyword arguments for extensibility.
        Returns:
            OutputType: Output from the pipeline.
        """
        pass

    def _get_node_inputs(self, node: PipelineNode) -> Dict[str, Any]:
        """
        Gather the input arguments for a pipeline node from the call trace.
        Args:
            node (PipelineNode): The pipeline node for which to gather inputs.
        Returns:
            Dict[str, Any]: Dictionary of input arguments for the node.
        """
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
        return input_kwargs

    def _handle_node_error(self, node: PipelineNode, error: Exception) -> bool:
        """
        Default error handling for node execution. Can be overridden by subclasses.
        Args:
            node (PipelineNode): The node where the error occurred.
            error (Exception): The exception raised during node execution.
        Returns:
            bool: True if the error should be skipped, False otherwise.
        """
        self.env.error_manager(self.call_trace, error)
        return False

    def _instanciate_nodes(self) -> Dict[str, Algorithm]:
        """
        Instantiate all pipeline nodes, filtering out those in disabled_qa if present.
        Returns:
            Dict[str, Algorithm]: Dictionary of node name to Algorithm instance.
        """
        instanciated_pipeline = self._instanciate_pipeline()
        nodes = {
            node.name: self._instanciate_node(
                node_class=node.algorithm.class_name,
                algorithm_params=node.algorithm.params,
                callbacks=node.callbacks,
            )
            for node in instanciated_pipeline
        }
        nodes = {
            node_name: node
            for node_name, node in nodes.items()
            if type(node) not in getattr(self.env, "disabled_qa", [])
        }
        return nodes

    def _instantiate_pipeline_node_param(self, param_value: Any) -> Any:
        """
        Instantiate a single parameter value if it has a class_name attribute.
        Args:
            param_value (Any): The parameter value to instantiate.
        Returns:
            Any: The instantiated parameter value.
        """
        if isinstance(param_value, (list, tuple)):
            # preserve the original type (list vs tuple)
            return type(param_value)(self._instantiate_pipeline_node_param(v) for v in param_value)
        if hasattr(param_value, "class_name"):
            return instantiate_class_from_name(class_name=param_value.class_name, kwargs=param_value.params)
        return param_value

    def _instanciate_pipeline(self) -> List[PipelineNode]:
        """
        Instantiate the pipeline nodes, resolving any nested PipelineClass references.
        Returns:
            List[PipelineNode]: List of instantiated pipeline nodes.
        """
        instanciated_pipeline = []
        for node in self.params.pipeline:
            current_node = node
            # Iterate over all algorithm parameters for the node and instantiate them if needed
            for param_name, param_value in node.algorithm.params.items():
                current_node.algorithm.params[param_name] = self._instantiate_pipeline_node_param(param_value)
            instanciated_pipeline.append(current_node)
        return instanciated_pipeline

    def _instanciate_node(
        self, node_class: str, algorithm_params: Dict[str, Any], callbacks: Optional[List[Any]]
    ) -> Algorithm:
        """
        Instantiate a single pipeline node, including its callbacks.
        Args:
            node_class (str): Fully qualified class name of the node.
            algorithm_params (Dict[str, Any]): Parameters for the node.
            callbacks (Optional[List[Any]]): Optional list of callback PipelineClass objects.
        Returns:
            Algorithm: Instantiated node.
        """
        if callbacks is not None and len(callbacks):
            instanciated_callbacks = [instantiate_class_from_name(cb.class_name, cb.params) for cb in callbacks]
            instanciated_callbacks = [
                cb for cb in instanciated_callbacks if type(cb) not in getattr(self.env, "disabled_qa", [])
            ]
            algorithm_params = {**algorithm_params, **{"callbacks": instanciated_callbacks}}
        return instantiate_class_from_name(node_class, algorithm_params)

    def _check_pipeline_coherency(self, params: Any) -> None:
        """
        Check the pipeline configuration for coherency (all node inputs must be declared prior to use).
        Args:
            params (Any): Pipeline parameters (should have a .pipeline attribute).
        Raises:
            Exception: If a node's input is not declared before its use.
        """
        parent_names = [PipelineCallTraceStorage.INPUT_KEY_NAME]
        for node in params.pipeline:
            for input_node in node.inputs:
                if isinstance(input_node.source_node, (tuple, list)):
                    for input_element in input_node.source_node:
                        if input_element.name not in parent_names:
                            raise Exception(
                                f"Pipeline configuration incoherent. Node {node.name} has input "
                                f"{input_element.name} not declared prior. Please fix pipeline configuration."
                            )
                elif input_node.source_node not in parent_names:
                    raise Exception(
                        f"Pipeline configuration incoherent. Node {node.name} has input "
                        f"{input_node.source_node} not declared prior. Please fix pipeline configuration."
                    )
            parent_names.append(node.name)
