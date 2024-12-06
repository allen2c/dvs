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


# Example Usage
if __name__ == "__main__":
    plaintext = "Hello, World!"
    base64_string = "SGVsbG8sIFdvcmxkIQ=="
    invalid_base64 = "SGVsbG8sIFdvcmxkIQ==="  # Invalid padding

    print(f"Is plaintext Base64? {is_base64(plaintext)}")  # Output: False
    print(f"Is Base64 string Base64? {is_base64(base64_string)}")  # Output: True
    print(f"Is invalid Base64 Base64? {is_base64(invalid_base64)}")  # Output: False

    # Additional Test Cases
    test_cases = [
        # Valid Base64 Strings
        ("Hello, World!", False),  # Plaintext
        ("SGVsbG8sIFdvcmxkIQ==", True),  # 'Hello, World!'
        ("SGVsbG8sIFdvcmxkIQ===", False),  # Invalid padding
        ("", False),  # Empty string
        ("Zg==", True),  # 'f'
        ("Zm8=", True),  # 'fo'
        ("Zm9v", True),  # 'foo'
        ("Zm9vYg==", True),  # 'foob'
        ("Zm9vYmE=", True),  # 'fooba'
        ("Zm9vYmFy", True),  # 'foobar'
        ("////", True),  # All valid Base64 characters
        ("AQIDBAUGBwgJCgsMDQ4PEA==", True),  # Binary data
        ("U29mdHdhcmUgRW5naW5lZXJpbmc=", True),  # 'Software Engineering'
        (
            "TWFuIGlzIGRpc3Rpbmd1aXNoZWQsIG5vdCBvbmx5IGJ5IGhpcyByZWFzb24=",
            True,
        ),  # Longer Base64 string
        ("VGhpcyBpcyBhbiBlbmNvZGVkIHN0cmluZy4=", True),  # 'This is an encoded string.'
        ("SGVsbG8gV29ybGQh", True),  # 'Hello World!'
        ("VGVzdCBzdHJpbmcgMTIzNDU=", True),  # 'Test string 12345'
        ("QkFTRTY0LUxvbmcgVGVzdA==", True),  # 'BASE64-Long Test'
        (
            "VGhpcyBpcyBhIHRlc3Qgd2l0aCBhbm90aGVyIGdyb3VuZCBlbGVtZW50",
            True,
        ),  # Without padding
        ("U3VwZXIgdmFsaWQgYmFzZTY0Lg==", True),  # 'Super valid base64.'
        ("VGVzdA==", True),  # 'Test'
        ("UHl0aG9u", True),  # 'Python'
        # Invalid Base64 Strings
        ("@@@=", False),  # Invalid characters
        ("Zm9v YmFy", False),  # Contains space
        ("Zm9v\tYmFy", False),  # Contains tab
        ("Zm9v\nYmFy", False),  # Contains newline
        ("Zm9vYmFy=", False),  # Incorrect padding
        ("Zm9vYmFy===", False),  # Excess padding
        ("A" * 5, False),  # Length not multiple of 4
        ("A A A A", False),  # Spaces between characters
        ("A@A@A@", False),  # Special characters
        ("Invalid_Base64!", False),  # Invalid characters
        ("!!!", False),  # Completely invalid
        ("Zg=", False),  # Invalid padding length
        ("Zg===", False),  # Too much padding
        ("Zg==Zg==", False),  # Multiple padding sections
        ("Zg=Zg==", False),  # Incorrect padding placement
    ]

    for i, (test_str, expected) in enumerate(test_cases, 1):
        result = is_base64(test_str)
        status = "PASS" if result == expected else "FAIL"
        print(
            f"Test Case {i}: {test_str!r} -> {result} (Expected: {expected}) [{status}]"
        )
        assert result == expected
