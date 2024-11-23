from collections import OrderedDict
from typing import List, Optional, Text, Union

import openai
from diskcache import Cache
from fastapi import HTTPException, status
from pydantic import BaseModel, Field

import dvs.utils.to as TO
from dvs.config import settings
from dvs.types.encoding_type import EncodingType


class SearchRequest(BaseModel):
    """
    Represents a single search request for vector similarity search.

    This class encapsulates the parameters needed to perform a vector similarity search
    in the DuckDB VSS API. It allows users to specify the query, the number of results
    to return, whether to include the embedding in the results, and the encoding type of the query.

    Attributes:
        query (Union[Text, List[float]]): The search query, which can be either a text string,
            a pre-computed vector embedding as a list of floats, or a base64 encoded string
            representing a vector. The interpretation of this field depends on the `encoding` attribute.
        top_k (int): The maximum number of results to return. Defaults to 5.
        with_embedding (bool): Whether to include the embedding vector in the search results.
            Defaults to False to reduce response size.
        encoding (Optional[EncodingType]): The encoding type for the query. Can be one of:
            - None (default): Assumes plaintext if query is a string, or a vector if it's a list of floats.
            - EncodingType.PLAINTEXT: Treats the query as plaintext to be converted to a vector.
            - EncodingType.BASE64: Treats the query as a base64 encoded vector.
            - EncodingType.VECTOR: Explicitly specifies that the query is a pre-computed vector.

    Example:
        >>> request = SearchRequest(query="How does AI work?", top_k=10, with_embedding=True)
        >>> print(request)
        SearchRequest(query='How does AI work?', top_k=10, with_embedding=True, encoding=None)

        >>> vector_request = SearchRequest(query=[0.1, 0.2, 0.3], top_k=5, encoding=EncodingType.VECTOR)
        >>> print(vector_request)
        SearchRequest(query=[0.1, 0.2, 0.3], top_k=5, with_embedding=False, encoding=<EncodingType.VECTOR: 'vector'>)

    Note:
        - When `encoding` is None or EncodingType.PLAINTEXT, and `query` is a string, it will be converted
          to a vector embedding using the configured embedding model.
        - When `encoding` is EncodingType.BASE64, the `query` should be a base64 encoded string
          representing a vector, which will be decoded before search.
        - When `encoding` is EncodingType.VECTOR or `query` is a list of floats, it's assumed
          to be a pre-computed embedding vector.
        - The `encoding` field provides flexibility for clients to send queries in different formats,
          allowing for optimization of request size and processing time.
    """  # noqa: E501

    query: Union[Text, List[float]] = Field(
        ...,
        description="The search query as text or a pre-computed vector embedding.",
    )
    top_k: int = Field(
        default=5,
        description="The maximum number of results to return.",
    )
    with_embedding: bool = Field(
        default=False,
        description="Whether to include the embedding in the result.",
    )
    encoding: Optional[EncodingType] = Field(
        default=None,
        description="The encoding type for the query.",
    )

    @classmethod
    async def to_vectors(
        cls,
        search_requests: "SearchRequest" | List["SearchRequest"],
        *,
        cache: Cache,
        openai_client: "openai.OpenAI",
    ) -> List[List[float]]:
        """
        Convert search requests to vector embeddings, handling various input types and encodings.

        This class method processes one or more SearchRequest objects, converting their queries
        into vector embeddings. It supports different input types (text, base64, or pre-computed vectors)
        and uses caching to improve performance for repeated queries.

        Parameters
        ----------
        search_requests : SearchRequest or List[SearchRequest]
            A single SearchRequest object or a list of SearchRequest objects to be processed.
        cache : Cache
            A diskcache.Cache object used for storing and retrieving cached embeddings.
        openai_client : openai.OpenAI
            An initialized OpenAI client object for making API calls to generate embeddings.

        Returns
        -------
        List[List[float]]
            A list of vector embeddings, where each embedding is a list of floats.
            The order of the output vectors corresponds to the order of the input search requests.

        Raises
        ------
        HTTPException
            If there's an error in processing any of the search requests, such as invalid encoding
            or mismatch between query type and encoding.

        Notes
        -----
        - The method handles three types of inputs:
        1. Text queries: Converted to embeddings using OpenAI's API (with caching).
        2. Base64 encoded vectors: Decoded to float vectors.
        3. Pre-computed float vectors: Used as-is.
        - For text queries, the method uses the `to_vectors_with_cache` function to generate
        and cache embeddings.
        - The method ensures that all output vectors have the correct dimensions as specified
        in the global settings.

        Examples
        --------
        >>> cache = Cache("./.cache/embeddings.cache")
        >>> openai_client = openai.OpenAI(api_key="your-api-key")
        >>> requests = [
        ...     SearchRequest(query="How does AI work?", top_k=5),
        ...     SearchRequest(query=[0.1, 0.2, 0.3, ...], top_k=3, encoding=EncodingType.VECTOR)
        ... ]
        >>> vectors = await SearchRequest.to_vectors(requests, cache=cache, openai_client=openai_client)
        >>> print(len(vectors), len(vectors[0]))
        2 512

        See Also
        --------
        to_vectors_with_cache : Function used for generating and caching text query embeddings.
        decode_base64_to_vector : Function used for decoding base64 encoded vectors.

        """  # noqa: E501

        search_requests = (
            [search_requests]
            if isinstance(search_requests, SearchRequest)
            else search_requests
        )

        output_vectors: List[Optional[List[float]]] = [None] * len(search_requests)
        required_emb_items: OrderedDict[int, Text] = OrderedDict()

        for idx, search_request in enumerate(search_requests):
            if not search_request.query:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid queries[{idx}].",
                )
            if isinstance(search_request.query, Text):
                if search_request.encoding == EncodingType.BASE64:
                    output_vectors[idx] = TO.base64_to_vector(search_request.query)
                elif search_request.encoding == EncodingType.VECTOR:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=(
                            f"Mismatch between queries[{idx}].encoding and "
                            + f"queries[{idx}].query."
                        ),
                    )
                else:
                    required_emb_items[idx] = search_request.query
            else:
                output_vectors[idx] = search_request.query

        # Ensure all required embeddings are text
        if len(required_emb_items) > 0:
            embeddings = await TO.queries_to_vectors_with_cache(
                list(required_emb_items.values()),
                cache=cache,
                openai_client=openai_client,
                model=settings.EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIMENSIONS,
            )
            for idx, embedding in zip(required_emb_items.keys(), embeddings):
                output_vectors[idx] = embedding

        # Ensure all vectors are not None
        for idx, v in enumerate(output_vectors):
            assert v is not None, f"output_vectors[{idx}] is None"
            assert (
                len(v) == settings.EMBEDDING_DIMENSIONS
            ), f"output_vectors[{idx}] has wrong dimensions"
        return output_vectors  # type: ignore
