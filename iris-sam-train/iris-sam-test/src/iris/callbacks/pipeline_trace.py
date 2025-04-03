from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from iris.callbacks.callback_interface import Callback
from iris.io.class_configs import Algorithm
from iris.orchestration.pipeline_dataclasses import PipelineNode


class PipelineCallTraceStorageError(Exception):
    """PipelineCallTraceStorage error class."""

    pass


class PipelineCallTraceStorage:
    """A storage object for pipeline input, intermediate and final results."""

    INPUT_KEY_NAME = "input"
    ERROR_KEY_NAME = "error"

    def __init__(self, results_names: Iterable[str]) -> None:
        """Assign parameters.

        Args:
            results_names (Iterable[str]): Create list of available keys in the storage.
        """
        self._storage = self._init_storage(results_names)

    def __getitem__(self, result_name: str) -> Any:
        """Get result_name result.

        Args:
            result_name (str): Result name.

        Raises:
            PipelineCallTraceStorageError: Raised if result_name is not found.

        Returns:
            Any: Result object.
        """
        return self.get(result_name)

    def __len__(self) -> int:
        """Get storage capacity.

        Returns:
            int: Storage capacity
        """
        return len(self._storage.keys())

    def get(self, result_name: str) -> Any:
        """Get result_name result.

        Args:
            result_name (str): Result name.

        Raises:
            PipelineCallTraceStorageError: Raised if result_name is not found.

        Returns:
            Any: Result object.
        """
        if result_name not in self._storage.keys():
            raise PipelineCallTraceStorageError(f"Unknown result name: {result_name}")

        return self._storage[result_name]

    def get_input(self) -> Any:
        """Return pipeline input.

        Returns:
            Any: Input to pipeline.
        """
        return self.get(PipelineCallTraceStorage.INPUT_KEY_NAME)

    def get_error(self) -> Optional[Exception]:
        """Return stored error.

        Returns:
            Optional[Exception]: error.
        """
        return self.get(PipelineCallTraceStorage.ERROR_KEY_NAME)

    def write(self, result_name: str, result: Any) -> None:
        """Write a result to a storage saved under the name `result_name`.

        Args:
            result_name (str): Result name.
            result (Any): Result reference to save.
        """
        self._storage[result_name] = result

    def write_input(self, in_value: Any) -> None:
        """Save `in_value` in storage.

        Args:
            in_value (Any): Input value.
        """
        self._storage[PipelineCallTraceStorage.INPUT_KEY_NAME] = in_value

    def write_error(self, error: Exception) -> None:
        """Save `error` in storage.

        Args:
            error (Exception): error to store.
        """
        self._storage[PipelineCallTraceStorage.ERROR_KEY_NAME] = error

    def clean(self) -> None:
        """Clean storage by setting all result references to None."""
        for result_name in self._storage.keys():
            self._storage[result_name] = None

    def _init_storage(self, results_names: Iterable[str]) -> Dict[str, None]:
        """Initialize storage (dict) with proper names and None values as results.

        Args:
            results_names (Iterable[str]): Result names.

        Returns:
            Dict[str, None]: Storage dictionary.
        """
        storage = {name: None for name in results_names}
        storage[PipelineCallTraceStorage.INPUT_KEY_NAME] = None
        storage[PipelineCallTraceStorage.ERROR_KEY_NAME] = None

        return storage

    @staticmethod
    def initialise(nodes: Dict[str, Algorithm], pipeline_nodes: List[PipelineNode]) -> PipelineCallTraceStorage:
        """Instantiate mechanisms for intermediate results tracing.

        Args:
            nodes (Dict[str, Algorithm]): Mapping between nodes names and the corresponding instanciated nodes.
            pipeline_nodes (List[PipelineNode]): List of nodes as declared in the input config. Not used in this function.

        Returns:
            PipelineCallTraceStorage: Pipeline intermediate and final results storage.
        """
        call_trace = PipelineCallTraceStorage(results_names=nodes.keys())

        for algorithm_name, algorithm_object in nodes.items():
            algorithm_object._callbacks.append(NodeResultsWriter(call_trace, algorithm_name))

        return call_trace


class NodeResultsWriter(Callback):
    """A node call results writer Callback class."""

    def __init__(self, trace_storage_reference: PipelineCallTraceStorage, result_name: str) -> None:
        """Assign parameters.

        Args:
            trace_storage_reference (PipelineCallTraceStorage): Storage object reference to write.
            result_name (str): Result name under which result should be written.
        """
        self._trace_storage_reference = trace_storage_reference
        self._result_name = result_name

    def on_execute_end(self, result: Any) -> None:
        """Write on node execution end.

        Args:
            result (Any): Result of node call.
        """
        self._trace_storage_reference.write(self._result_name, result)
