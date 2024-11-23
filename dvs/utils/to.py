import asyncio
import base64
import binascii
from typing import List, Optional, Text, Union

import numpy as np
import openai
from diskcache import Cache
from fastapi import HTTPException, status

from dvs.config import settings


async def queries_to_vectors_with_cache(
    queries: Union[List[Text], Text],
    *,
    cache: Cache,
    openai_client: "openai.OpenAI",
    model: Text,
    dimensions: int,
) -> List[List[float]]:
    """
    Convert input queries to vector embeddings using OpenAI's API, with caching.

    This function takes a list of text queries or a single text query and converts them
    into vector embeddings. It uses a cache to store and retrieve previously computed
    embeddings, reducing API calls and improving performance for repeated queries.

    Parameters
    ----------
    queries : Union[List[Text], Text]
        A single text query or a list of text queries to be converted into vector embeddings.
    cache : Cache
        A diskcache.Cache object used for storing and retrieving cached embeddings.
    openai_client : openai.OpenAI
        An initialized OpenAI client object for making API calls.

    Returns
    -------
    List[List[float]]
        A list of vector embeddings, where each embedding is a list of floats.

    Raises
    ------
    ValueError
        If the function fails to get embeddings for all queries.

    Notes
    -----
    - The function first checks the cache for each query. If found, it uses the cached embedding.
    - For queries not in the cache, it batches them and sends a single request to the OpenAI API.
    - New embeddings are cached with an expiration time of 7 days (604800 seconds).
    - The embedding model and dimensions are determined by the global `settings` object.

    Example
    -------
    >>> cache = Cache("./.cache/embeddings.cache")
    >>> openai_client = openai.OpenAI(api_key="your-api-key")
    >>> queries = ["How does AI work?", "What is machine learning?"]
    >>> embeddings = await to_vectors_with_cache(queries, cache=cache, openai_client=openai_client)
    >>> print(len(embeddings), len(embeddings[0]))
    2 512

    See Also
    --------
    ensure_vectors : A higher-level function that handles various input types and uses this function.
    """  # noqa: E501

    queries = [queries] if isinstance(queries, Text) else queries
    output_vectors: List[Optional[List[float]]] = [None] * len(queries)
    not_cached_indices: List[int] = []
    for idx, query in enumerate(queries):
        cached_vector = await asyncio.to_thread(cache.get, query)
        if cached_vector is None:
            not_cached_indices.append(idx)
        else:
            output_vectors[idx] = cached_vector  # type: ignore

    # Get embeddings for queries that are not cached
    if not_cached_indices:
        not_cached_queries = [queries[i] for i in not_cached_indices]
        embeddings_response = await asyncio.to_thread(
            openai_client.embeddings.create,
            input=not_cached_queries,
            model=model,
            dimensions=dimensions,
        )
        embeddings_data = embeddings_response.data
        for idx, embedding in zip(not_cached_indices, embeddings_data):
            await asyncio.to_thread(
                cache.set, queries[idx], embedding.embedding, expire=604800
            )
            output_vectors[idx] = embedding.embedding  # type: ignore

    if any(v is None for v in output_vectors):
        raise ValueError("Failed to get embeddings for all queries")
    return output_vectors  # type: ignore


def base64_to_vector(base64_str: Text) -> List[float]:
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

    try:
        vector = np.frombuffer(  # type: ignore[no-untyped-call]
            base64.b64decode(base64_str), dtype="float32"
        ).tolist()
        if len(vector) != settings.EMBEDDING_DIMENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid embedding dimensions: {len(vector)}",
            )
        return vector
    except (binascii.Error, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid base64 string: {base64_str}",
        )
