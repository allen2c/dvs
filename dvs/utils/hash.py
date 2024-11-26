import hashlib
from typing import Text


def hash_content(content: Text) -> Text:
    return hashlib.md5(content.strip().encode("utf-8")).hexdigest()
