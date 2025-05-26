import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml

from iris.io.class_configs import Algorithm, instantiate_class_from_name
from iris.io.dataclasses import IrisTemplate
from iris.orchestration.pipeline_dataclasses import PipelineClass

# from iris.io.dataclasses import IrisTemplate


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


class MultiframeAggregation(Algorithm):
    class Parameters(Algorithm.Parameters):
        metadata: Dict[str, Any]
        algorithm: PipelineClass

    __parameters_type__ = Parameters

    def __init__(self, metadata: Dict[str, Any], algorithm: PipelineClass, **kwargs):
        super().__init__(metadata=metadata, algorithm=algorithm, **kwargs)
        self.algorithm = instantiate_class_from_name(algorithm["class_name"], algorithm["params"])

    @classmethod
    def from_config(cls, config: Optional[str]):
        deserialized_config = _load_yaml_config(
            config, os.path.join(os.path.dirname(__file__), "confs", "multiframe_pipeline.yaml")
        )
        return cls(metadata=deserialized_config["metadata"], algorithm=deserialized_config["algorithm"])

    @classmethod
    def from_irispipeline_config(cls, config: Optional[str], keyword: str = "templates_aggregation"):
        deserialized_config = _load_yaml_config(
            config, os.path.join(os.path.dirname(__file__), "confs", "multiframe_pipeline.yaml")
        )

        if keyword not in deserialized_config:
            raise ValueError(
                f"MultiframeAggregation requires a valid keyword in the configuration file. Please check the keyword: {keyword}"
            )

        aggregation_config = deserialized_config[keyword]
        return cls(metadata=aggregation_config["metadata"], algorithm=aggregation_config["algorithm"])

    def run(self, templates: List[IrisTemplate], **kwargs) -> Tuple[IrisTemplate, np.ndarray]:
        combined_template, weights = self.algorithm.run(templates)
        return combined_template, weights

    # def instanciate_algorithm(
    #     self, node_class: str, algorithm_params: Dict[str, Any], callbacks: Optional[List[PipelineClass]]
    # ) -> Algorithm:
    #     """Instanciate an Algorithm from its class, kwargs and optional Callbacks.

    #     NOTE: All callbacks of type listed in self.env.disabled_qa will be filtered out. This allows one config file to be used in various QA standards levels.

    #     Args:
    #         node_class (str): Node's class.
    #         algorithm_params (Dict[str, Any]): Node's kwargs.
    #         callbacks (Optional[List[PipelineClass]]): list of callbacks.

    #     Returns:
    #         Algorithm: instanciated node.
    #     """
    #     if callbacks is not None and len(callbacks):
    #         instanciated_callbacks = [self.instanciate_class(cb.class_name, cb.params) for cb in callbacks]
    #         instanciated_callbacks = [cb for cb in instanciated_callbacks if type(cb) not in self.env.disabled_qa]

    #         algorithm_params = {**algorithm_params, **{"callbacks": instanciated_callbacks}}

    #     return self.instanciate_class(node_class, algorithm_params)
