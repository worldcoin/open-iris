from typing import Any, Dict, List, Optional, Union

import numpy as np

from iris.io.class_configs import ImmutableModel
from iris.orchestration.environment import Environment
from iris.pipelines.iris_pipeline import IRISPipeline
from iris.pipelines.multiframe_aggregation_pipeline import MultiframeAggregationPipeline


class MultiframeIRISPipeline:
    """
    Runs IRISPipeline for N frames and aggregates the results using MultiframeAggregationPipeline.
    """

    class Parameters(ImmutableModel):
        iris_pipeline_config: Union[Dict[str, Any], Optional[str]] = None
        aggregation_pipeline_config: Union[Dict[str, Any], Optional[str]] = None
        iris_env: Optional[Environment] = None
        aggregation_env: Optional[Environment] = None

    def __init__(self, params: Optional[Union[Parameters, Dict[str, Any]]] = None, **kwargs):
        """
        Args:
            params: Parameters object or dict with keys iris_pipeline_config, aggregation_pipeline_config, iris_env, aggregation_env
            kwargs: For convenience, can also pass parameters directly as kwargs
        """
        if params is None:
            params = {}
        if isinstance(params, dict):
            params = {**params, **kwargs}
            self.params = self.Parameters(**params)
        else:
            self.params = params

        self.iris_pipeline = IRISPipeline(
            config=self.params.iris_pipeline_config, env=self.params.iris_env if self.params.iris_env else None
        )
        self.aggregation_pipeline = MultiframeAggregationPipeline(
            config=self.params.aggregation_pipeline_config,
            env=self.params.aggregation_env if self.params.aggregation_env else None,
        )

    @property
    def iris_call_trace(self):
        """Access the call_trace of the last IRISPipeline run (for debugging)."""
        return self.iris_pipeline.call_trace

    @property
    def aggregation_call_trace(self):
        """Access the call_trace of the last MultiframeAggregationPipeline run (for debugging)."""
        return self.aggregation_pipeline.call_trace

    def run(self, images: List[np.ndarray], eye_side: str) -> Any:
        """
        Run IRISPipeline for each image, then aggregate the results.

        Args:
            images (List[np.ndarray]): List of IR images.
            eye_side (str): 'left' or 'right'.

        Returns:
            Any: Output of the aggregation pipeline (typically a dict with iris_template, error, metadata, etc.)
        """
        templates = []
        for img in images:
            iris_output = self.iris_pipeline.run(img, eye_side)
            # Expecting iris_output to be a dict with 'iris_template' key, or directly an IrisTemplate
            if isinstance(iris_output, dict) and "iris_template" in iris_output:
                template = iris_output["iris_template"]
            else:
                template = iris_output
            if template is not None:
                templates.append(template)
        return self.aggregation_pipeline.run(templates)
