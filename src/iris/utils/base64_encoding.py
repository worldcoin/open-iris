import base64
from typing import Tuple

import numpy as np


def base64_encode_array(array2encode: np.ndarray) -> bytes:
    """Convert a numpy array to a packed base64 string.

    Args:
        array2encode (np.ndarray): The array to convert.

    Returns:
        bytes: The packed base64 string.
    """
    co_pack = np.packbits(array2encode)

    return base64.b64encode(co_pack.tobytes())


def base64_decode_array(bytes_array: str, array_shape: Tuple[int, int, int, int] = (16, 256, 2, 2)) -> np.ndarray:
    """Convert a packed base64 string to a numpy array.

    Args:
        bytes_array (bytes): The packed base64 byte string.
        shape (Tuple[int, int, int, int], optional): The shape of the array. Defaults to (16, 256, 2, 2).

    Returns:
        np.ndarray: The array.
    """
    decoded_bytes = base64.b64decode(bytes_array)

    deserialized_bytes = np.frombuffer(decoded_bytes, dtype=np.uint8)
    unpacked_bits = np.unpackbits(deserialized_bytes)

    return unpacked_bits.reshape(*array_shape).astype(bool)


def base64_encode_str(input_str: str) -> str:
    """Convert a string to base64 string. Both input and output are string, but base64 encoded vs non-encoded.

    Args:
        input_str (str): The string to encode.

    Returns:
        str: the encoded base64 string.
    """
    return base64.b64encode(input_str.encode()).decode()


def base64_decode_str(base64_str: str) -> str:
    """Convert base64-encoded string to decoded string. Both input and output are string, but base64 encoded vs non-encoded.

    Args:
        base64_str (str): The base64-encoded string

    Returns:
        str: the decoded string
    """
    return base64.b64decode(base64_str).decode()
