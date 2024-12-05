from dvs.config import settings
from dvs.utils.cache import get_embedding_cache_key
from dvs.utils.hash import hash_content


def test_get_embedding_cache_key():
    """Test cache key generation with default parameters"""

    content = "test_content"
    result = get_embedding_cache_key(content)

    assert (
        result
        == f"cache:{settings.EMBEDDING_MODEL}:{settings.EMBEDDING_DIMENSIONS}:{hash_content(content)}"  # noqa: E501
    )
