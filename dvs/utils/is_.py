import base64
import binascii
import re


def is_base64(s: str) -> bool:
    """
    Determines whether a given string is a valid Base64 encoded string.

    This function checks if the input string meets the criteria for Base64 encoding, including length and character validity. It attempts to decode the string using Base64 decoding and returns `True` if successful, or `False` if any checks fail or an exception occurs during decoding.

    Note:
    - The input string must have a length that is a multiple of 4.
    - The function uses a regular expression to validate the characters in the string.
    - It handles exceptions related to invalid Base64 strings during the decoding process.

    Examples:
    - `is_base64("SGVsbG8sIFdvcmxkIQ==")` returns `True`.
    - `is_base64("Invalid_Base64!")` returns `False`.
    """  # noqa: E501

    # Base64 strings should have a length that's a multiple of 4
    if len(s) % 4 != 0:
        return False

    # Regular expression to match valid Base64 characters
    base64_regex = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")
    if not base64_regex.match(s):
        return False

    try:
        # Attempt to decode the string
        decoded_bytes = base64.b64decode(s, validate=True)  # noqa: F841
        # Optionally, you can add more checks here to verify the decoded bytes
        return True
    except (binascii.Error, ValueError):
        return False
