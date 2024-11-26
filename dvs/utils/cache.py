from typing import Text

from dvs.config import settings


def get_embedding_cache_key(
    content: Text,
    model: Text = settings.EMBEDDING_MODEL,
    dimensions: int = settings.EMBEDDING_DIMENSIONS,
) -> Text:
    from dvs.utils.hash import hash_content

    return f"cache:{model}:{dimensions}:{hash_content(content)}"
