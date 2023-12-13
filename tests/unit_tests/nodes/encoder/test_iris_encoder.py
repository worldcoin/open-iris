import pytest
from pydantic import ValidationError

from iris.nodes.encoder.iris_encoder import IrisEncoder


@pytest.mark.parametrize(
    "mask_threshold",
    [pytest.param(-0.5), pytest.param(1.5)],
    ids=[
        "mask_threshold should not be negative",
        "mask_threshold should not be larger than 1",
    ],
)
def test_iris_encoder_threshold_raises_an_exception(
    mask_threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        _ = IrisEncoder(mask_threshold)
