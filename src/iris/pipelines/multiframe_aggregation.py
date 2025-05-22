from typing import Any, Dict, Iterator

from iris.io.dataclasses import IrisTemplate
from iris.multiframe.iris_multiframe_combiner import TemplateCombinerFactory


class MultiframeAggregation:
    def __init__(self, strategy: str = "majority_vote", **kwargs):
        self.combiner = TemplateCombinerFactory.create(strategy, **kwargs)

    @classmethod
    def load_from_config(cls, config: Dict[str, Any]):
        return cls(strategy=config["strategy"], **config["parameters"])

    def __call__(self, templates: Iterator[IrisTemplate], **kwargs):
        # Combine templates
        combined_template, weights = self.combiner.combine_templates(templates)
        # Optionally, store or return weights as needed
        return combined_template, weights
