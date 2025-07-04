import base64
import typing

import numpy as np


def base64_to_vector(base64_str: typing.Text) -> typing.List[float]:
    """
    Decode a base64 encoded string to a vector of floats.

    This function attempts to decode a base64 encoded string into a vector of
    float values. It's particularly useful for converting encoded embeddings
    back into their original numerical representation.

    Parameters
    ----------
    base64_str : Text
        A string containing the base64 encoded vector data.

    Returns
    -------
    Optional[List[float]]
        If decoding is successful, returns a list of float values representing
        the vector. If decoding fails, returns None.

    Notes
    -----
    The function uses numpy to interpret the decoded bytes as a float32 array
    before converting it to a Python list. This approach is efficient for
    handling large vectors.

    The function is designed to gracefully handle decoding errors, returning
    None instead of raising an exception if the input is not a valid base64
    encoded string or cannot be interpreted as a float32 array.

    Examples
    --------
    >>> encoded = "AAAAAAAAAEA/AABAQAAAQUA="
    >>> result = decode_base64_to_vector(encoded)
    >>> print(result)
    [0.0, 0.5, 1.0, 1.5]

    >>> invalid = "Not a base64 string"
    >>> result = decode_base64_to_vector(invalid)
    >>> print(result)
    None

    See Also
    --------
    base64.b64decode : For decoding base64 strings.
    numpy.frombuffer : For creating numpy arrays from buffer objects.
    """  # noqa: E501
    vector = np.frombuffer(base64.b64decode(base64_str), dtype="float32").tolist()
    return vector


def vector_to_base64(vector: typing.List[float]) -> typing.Text:
    """
    Convert a list of floats to a base64 encoded string.

    This function takes a list of float values, converts it to a NumPy array of type float32,
    and then encodes the byte representation of the array into a base64 string. This is useful
    for transmitting or storing vector data in a compact format.

    Examples
    --------
    >>> vector = [0.1, 0.2, 0.3]
    >>> encoded = vector_to_base64(vector)
    >>> print(encoded)
    'AAECAwQ='
    """  # noqa: E501

    base64_bytes = base64.b64encode(np.array(vector, dtype=np.float32).tobytes())
    return base64_bytes.decode("ascii")
