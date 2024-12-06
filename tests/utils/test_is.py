import typing

import pytest

from dvs.utils.is_ import is_base64
from dvs.utils.to import vector_to_base64


@pytest.mark.parametrize(
    ("input", "expected"),
    [
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
        (vector_to_base64([0.32, 0.25]), True),  # Numpy array 1-D
    ],
)
def test_is_base64(input: typing.Text, expected: bool):
    assert is_base64(input) == expected
