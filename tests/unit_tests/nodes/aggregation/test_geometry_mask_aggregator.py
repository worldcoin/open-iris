from typing import List

import numpy as np
import pytest

from iris.io.dataclasses import NoiseMask
from iris.nodes.aggregation.noise_mask_union import NoiseMaskUnion


def array_to_NoiseMask(input_array: list) -> NoiseMask:
    return NoiseMask(mask=np.array(input_array).astype(bool))


@pytest.mark.parametrize(
    "geometry_masks,expected_output",
    [
        ([array_to_NoiseMask([[0, 1], [1, 0]])], array_to_NoiseMask([[0, 1], [1, 0]])),
        (
            [array_to_NoiseMask([[0, 0, 1], [1, 0, 0]]), array_to_NoiseMask([[0, 0, 0], [1, 1, 0]])],
            array_to_NoiseMask([[0, 0, 1], [1, 1, 0]]),
        ),
    ],
    ids=["1 NoiseMask", "2 NoiseMasks"],
)
def test_geometry_mask_aggregate_mask(geometry_masks: List[NoiseMask], expected_output: NoiseMask) -> None:
    aggregation_node = NoiseMaskUnion()

    aggregated_geometry_mask = aggregation_node.execute(geometry_masks)

    np.testing.assert_equal(aggregated_geometry_mask.mask, expected_output.mask)
