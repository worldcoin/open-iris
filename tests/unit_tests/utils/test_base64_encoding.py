import numpy as np
import pytest

from iris.utils import base64_encoding as be


@pytest.mark.parametrize("mock_shape", [(3, 10, 100), (10, 3, 100), (100, 10, 3)])
def test_base64_array_encode_decode(mock_shape: tuple) -> None:
    mock_array = np.random.choice(2, size=mock_shape).astype(bool)

    result = be.base64_decode_array(be.base64_encode_array(mock_array), array_shape=mock_shape)

    np.testing.assert_equal(result, mock_array)


@pytest.mark.parametrize(
    "plain_str,base64_str", [("test", "dGVzdA=="), ("un:\n  - deux\n  - trois", "dW46CiAgLSBkZXV4CiAgLSB0cm9pcw==")]
)
def test_base64_str_encode_decode(plain_str: str, base64_str: str) -> None:
    # Test base64_encode_str
    encoded_str = be.base64_encode_str(plain_str)
    assert encoded_str == base64_str
    assert isinstance(encoded_str, str)

    # Test base64_decode_str
    decoded_str = be.base64_decode_str(base64_str)
    assert decoded_str == plain_str
    assert isinstance(decoded_str, str)

    # Test that encoding and decoding convolve
    encoded_decoded_str = be.base64_decode_str(be.base64_encode_str(plain_str))
    assert encoded_decoded_str == plain_str
