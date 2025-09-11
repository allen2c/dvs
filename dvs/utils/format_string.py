def format_string(s: str) -> str:
    return s.casefold().replace("-", "_").replace(" ", "_").strip()


def sanitize_xml_string(s: str) -> str:
    """
    Sanitize a string to be safe for XML by removing control characters
    and other invalid characters that can cause XML parsing errors.

    Args:
        s: The string to sanitize

    Returns:
        A sanitized string safe for XML content
    """
    if not s:
        return s

    # Remove control characters (0x00-0x1F and 0x7F-0x9F) except for tab, LF, CR
    # These are invalid in XML and can cause parsing errors
    valid_chars = []
    for char in s:
        code = ord(char)
        # Allow: tab (9), LF (10), CR (13)
        # Allow: printable characters (32-126)
        # Allow: extended characters (128+)
        if (code >= 32 and code <= 126) or code >= 128 or code in (9, 10, 13):
            valid_chars.append(char)
        # Replace invalid characters with a safe placeholder
        else:
            valid_chars.append("?")  # Could also use '' to remove them entirely

    return "".join(valid_chars)
