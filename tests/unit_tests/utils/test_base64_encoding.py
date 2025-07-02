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


@pytest.mark.parametrize(
    "mock_shape,dtype",
    [
        ((3, 10, 100), np.float32),
        ((100, 10, 3), np.float32),
    ],
)
def test_base64_float_array_encode_decode(mock_shape, dtype):
    mock_array = np.random.rand(*mock_shape).astype(dtype)

    # Encode and decode
    b64 = be.base64_encode_float_array(mock_array)
    decoded = be.base64_decode_float_array(b64, mock_shape, dtype=dtype)

    # Use allclose for floats
    np.testing.assert_allclose(decoded, mock_array, rtol=1e-6, atol=1e-6)
    assert decoded.dtype == mock_array.dtype
