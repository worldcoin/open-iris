from typing import Any, Dict

from iris.io.class_configs import Algorithm

# from iris.io.dataclasses import IrisTemplate


class MultiframeAggregation(Algorithm):
    class Parameters(Algorithm.Parameters):
        metadata: Dict[str, Any]
        algorithm: Dict[str, Any]

    __parameters_type__ = Parameters

    def __init__(self, metadata: Dict[str, Any], algorithm: Dict[str, Any], **kwargs):
        super().__init__(metadata=metadata, algorithm=algorithm, **kwargs)
        # self.algorithm =

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
