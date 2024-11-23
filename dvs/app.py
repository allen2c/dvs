"""
DVS - DuckDB Vector Similarity Search (VSS) API

This module implements a FastAPI-based web service for performing vector similarity searches
using DuckDB as the backend database and OpenAI's embedding models for vector representation.

The API provides endpoints for single and bulk vector similarity searches, allowing users to
find similar documents or data points based on text queries or pre-computed vector embeddings.

Key Components:
---------------
1. FastAPI Application: Handles HTTP requests and responses.
2. DuckDB Integration: Utilizes DuckDB for efficient storage and querying of vector data.
3. OpenAI Embedding: Converts text queries into vector embeddings using OpenAI's models.
4. Caching: Implements a disk-based cache to store and retrieve computed embeddings.
5. Vector Search: Performs cosine similarity-based searches on the vector database.

Main Endpoints:
---------------
- GET /: Root endpoint providing API status and information.
- POST /search or /s: Perform a single vector similarity search.
- POST /bulk_search or /bs: Perform multiple vector similarity searches in one request.

Configuration:
--------------
The API behavior is controlled by environment variables and the `Settings` class,
which includes database paths, table names, embedding model details, and cache settings.

Data Models:
------------
- Point: Represents a point in the vector space.
- Document: Contains metadata and content information for documents.
- SearchRequest: Defines parameters for a single search request.
- BulkSearchRequest: Represents multiple search requests in a single call.
- SearchResult: Contains the result of a single vector similarity search.
- SearchResponse: Wraps multiple SearchResults for API responses.

Usage Flow:
-----------
1. Client sends a search request (text or vector) to the API.
2. API converts text to vector embedding if necessary (using OpenAI API).
3. Vector search is performed on the DuckDB database.
4. Results are processed and returned to the client.

Performance Considerations:
---------------------------
- Caching of embeddings reduces API calls and improves response times.
- Bulk search endpoint allows for efficient processing of multiple queries.
- DuckDB's columnar storage and vector operations ensure fast similarity computations.

Dependencies:
-------------
- FastAPI: Web framework for building APIs.
- DuckDB: Embeddable SQL OLAP database management system.
- OpenAI: For generating text embeddings.
- Pydantic: Data validation and settings management.
- NumPy: For numerical operations on vectors.

For detailed API documentation, refer to the OpenAPI schema available at the /docs endpoint.

"""  # noqa: E501

import asyncio
import time
from collections import OrderedDict
from textwrap import dedent
from typing import Dict, List, Optional, Text, Tuple, Union

import duckdb
import openai
from diskcache import Cache
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field

import dvs.utils.to as TO
import dvs.utils.vss as VSS
from dvs.config import settings
from dvs.types.document import Document
from dvs.types.encoding_type import EncodingType
from dvs.types.point import Point


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


class BulkSearchRequest(BaseModel):
    """
    Represents a bulk search request for multiple vector similarity searches.

    This class allows users to submit multiple search queries in a single API call,
    which can be more efficient than making separate requests for each query.

    Attributes:
        queries (List[SearchRequest]): A list of SearchRequest objects, each representing
            an individual search query with its own parameters.

    Example:
        >>> bulk_request = BulkSearchRequest(queries=[
        ...     SearchRequest(query="How does AI work?", top_k=5),
        ...     SearchRequest(query="What is machine learning?", top_k=3, with_embedding=True)
        ... ])
        >>> print(bulk_request)
        BulkSearchRequest(queries=[SearchRequest(query='How does AI work?', top_k=5, with_embedding=False), SearchRequest(query='What is machine learning?', top_k=3, with_embedding=True)])

    Note:
        The bulk search functionality allows for efficient processing of multiple queries
        in parallel, which can significantly reduce overall response time compared to
        sequential individual requests.
    """  # noqa: E501

    queries: List[SearchRequest] = Field(
        ...,
        description="A list of search requests to be processed in bulk.",
    )


class SearchResult(BaseModel):
    """
    Represents a single result from a vector similarity search operation.

    This class encapsulates the information returned for each matching item
    in a vector similarity search, including the matched point, its associated
    document (if any), and the relevance score indicating how closely it matches
    the query.

    Attributes
    ----------
    point : Point
        The matched point in the vector space, containing embedding and metadata.
    document : Optional[Document]
        The associated document for the matched point, if available.
    relevance_score : float
        A score indicating the similarity between the query and the matched point,
        typically ranging from 0 to 1, where 1 indicates a perfect match.

    Methods
    -------
    from_search_result(search_result: Tuple[Point, Optional[Document], float]) -> SearchResult
        Class method to create a SearchResult instance from a tuple of search results.

    Notes
    -----
    The relevance score is typically calculated using cosine similarity between
    the query vector and the point's embedding vector.

    Examples
    --------
    >>> point = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
    >>> document = Document(document_id="doc1", name="Example Doc", content="Sample content")
    >>> result = SearchResult(point=point, document=document, relevance_score=0.95)
    >>> print(result.relevance_score)
    0.95
    """  # noqa: E501

    point: Point = Field(
        ...,
        description="The matched point in the vector space.",
    )
    document: Optional[Document] = Field(
        default=None,
        description="The associated document for the matched point, if available.",
    )
    relevance_score: float = Field(
        default=0.0,
        description="The similarity score between the query and the matched point.",
    )

    @classmethod
    def from_search_result(
        cls, search_result: Tuple["Point", Optional["Document"], float]
    ) -> "SearchResult":
        """
        Create a SearchResult instance from a tuple of search results.

        Parameters
        ----------
        search_result : Tuple[Point, Optional[Document], float]
            A tuple containing the point, document, and relevance score.

        Returns
        -------
        SearchResult
            An instance of SearchResult created from the input tuple.

        Examples
        --------
        >>> point = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
        >>> document = Document(document_id="doc1", name="Example Doc", content="Sample content")
        >>> result_tuple = (point, document, 0.95)
        >>> search_result = SearchResult.from_search_result(result_tuple)
        >>> print(search_result.relevance_score)
        0.95
        """  # noqa: E501

        return cls.model_validate(
            {
                "point": search_result[0],
                "document": search_result[1],
                "relevance_score": search_result[2],
            }
        )


class SearchResponse(BaseModel):
    """
    Represents the response to a single vector similarity search query.

    This class encapsulates a list of SearchResult objects, providing a
    structured way to return multiple matching results for a given query.

    Attributes
    ----------
    results : List[SearchResult]
        A list of SearchResult objects, each representing a matched item
        from the vector similarity search.

    Methods
    -------
    from_search_results(search_results: List[Tuple[Point, Optional[Document], float]]) -> SearchResponse
        Class method to create a SearchResponse instance from a list of search result tuples.

    Notes
    -----
    The results are typically ordered by relevance score in descending order,
    with the most similar matches appearing first in the list.

    Examples
    --------
    >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
    >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
    >>> result1 = SearchResult(point=point1, document=document1, relevance_score=0.95)
    >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
    >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
    >>> result2 = SearchResult(point=point2, document=document2, relevance_score=0.85)
    >>> response = SearchResponse(results=[result1, result2])
    >>> print(len(response.results))
    2
    """  # noqa: E501

    results: List[SearchResult] = Field(
        default_factory=list,
        description="A list of search results from the vector similarity search.",
    )

    @classmethod
    def from_search_results(
        cls,
        search_results: List[Tuple["Point", Optional["Document"], float]],
    ) -> "SearchResponse":
        """
        Create a SearchResponse instance from a list of search result tuples.

        Parameters
        ----------
        search_results : List[Tuple[Point, Optional[Document], float]]
            A list of tuples, each containing a point, an optional document,
            and a relevance score.

        Returns
        -------
        SearchResponse
            An instance of SearchResponse created from the input list of tuples.

        Examples
        --------
        >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
        >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
        >>> result1 = (point1, document1, 0.95)
        >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
        >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
        >>> result2 = (point2, document2, 0.85)
        >>> response = SearchResponse.from_search_results([result1, result2])
        >>> print(len(response.results))
        2
        """  # noqa: E501

        return cls.model_validate(
            {
                "results": [
                    SearchResult.from_search_result(res) for res in search_results
                ]
            }
        )


class BulkSearchResponse(BaseModel):
    """
    Represents the response to a bulk vector similarity search operation.

    This class encapsulates a list of SearchResponse objects, each corresponding
    to a single query in a bulk search request. It provides a structured way to
    return results for multiple queries in a single response.

    Attributes
    ----------
    results : List[SearchResponse]
        A list of SearchResponse objects, each containing the results for
        a single query in the bulk search operation.

    Methods
    -------
    from_bulk_search_results(bulk_search_results: List[List[Tuple[Point, Optional[Document], float]]]) -> BulkSearchResponse
        Class method to create a BulkSearchResponse instance from a list of bulk search result tuples.

    Notes
    -----
    The order of SearchResponse objects in the results list corresponds to
    the order of queries in the original bulk search request.

    Examples
    --------
    >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
    >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
    >>> result1 = SearchResult(point=point1, document=document1, relevance_score=0.95)
    >>> response1 = SearchResponse(results=[result1])
    >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
    >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
    >>> result2 = SearchResult(point=point2, document=document2, relevance_score=0.85)
    >>> response2 = SearchResponse(results=[result2])
    >>> bulk_response = BulkSearchResponse(results=[response1, response2])
    >>> print(len(bulk_response.results))
    2
    """  # noqa: E501

    results: List[SearchResponse] = Field(
        default_factory=list,
        description="A list of search responses, each corresponding to a query in the bulk search.",  # noqa: E501
    )

    @classmethod
    def from_bulk_search_results(
        cls,
        bulk_search_results: List[List[Tuple["Point", Optional["Document"], float]]],
    ) -> "BulkSearchResponse":
        """
        Create a BulkSearchResponse instance from a list of bulk search result tuples.

        Parameters
        ----------
        bulk_search_results : List[List[Tuple[Point, Optional[Document], float]]]
            A list of lists, where each inner list contains tuples of search results
            for a single query in the bulk search operation.

        Returns
        -------
        BulkSearchResponse
            An instance of BulkSearchResponse created from the input list of bulk search results.

        Examples
        --------
        >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
        >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
        >>> result1 = [(point1, document1, 0.95)]
        >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
        >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
        >>> result2 = [(point2, document2, 0.85)]
        >>> bulk_response = BulkSearchResponse.from_bulk_search_results([result1, result2])
        >>> print(len(bulk_response.results))
        2
        """  # noqa: E501

        return cls.model_validate(
            {
                "results": [
                    SearchResponse.from_search_results(search_results)
                    for search_results in bulk_search_results
                ]
            }
        )


# FastAPI app
app = FastAPI(
    debug=False if settings.APP_ENV == "production" else True,
    title=settings.APP_NAME,
    summary="A high-performance vector similarity search API powered by DuckDB and OpenAI embeddings",  # noqa: E501
    description=dedent(
        """
        DVS - The DuckDB Vector Similarity Search (VSS) API provides a fast and efficient way to perform
        vector similarity searches on large datasets. It leverages DuckDB for data storage and
        querying, and OpenAI's embedding models for vector representation of text data.

        Key Features:
        - Single and bulk vector similarity searches
        - Caching of embeddings for improved performance
        - Support for both text queries and pre-computed vector embeddings
        - Configurable search parameters (e.g., top-k results, embedding inclusion)
        - Integration with OpenAI's latest embedding models

        This API is designed for applications requiring fast similarity search capabilities,
        such as recommendation systems, semantic search engines, and content discovery platforms.
        """  # noqa: E501
    ).strip(),
    version=settings.APP_VERSION,
    contact={
        "name": "DuckDB VSS API",
        "url": "https://github.com/allen2c/dvs.git",
        "email": "f1470891079@gmail.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)
# FastAPI app state
app.state.settings = app.extra["settings"] = settings
app.state.cache = app.extra["cache"] = Cache(
    directory=settings.CACHE_PATH, size_limit=settings.CACHE_SIZE_LIMIT
)
# OpenAI client
if settings.OPENAI_API_KEY is None:
    app.state.openai_client = app.extra["openai_client"] = None
else:
    app.state.openai_client = app.extra["openai_client"] = openai.OpenAI(
        api_key=settings.OPENAI_API_KEY
    )

# DuckDB connection
with duckdb.connect(settings.DUCKDB_PATH) as __conn__:
    __conn__.sql("INSTALL json;")
    __conn__.sql("LOAD json;")
    __conn__.sql("INSTALL vss;")
    __conn__.sql("LOAD vss;")


# API endpoints
@app.get("/", description="Root endpoint providing API status and basic information")
async def api_root() -> Dict[Text, Text]:
    """
    Root endpoint for the DuckDB Vector Similarity Search (VSS) API.

    This endpoint serves as the entry point for the API, providing basic
    information about the API's status and version. It can be used for
    health checks, API discovery, or as a starting point for API exploration.

    Returns
    -------
    dict
        A dictionary containing status information and API details.
        {
            "status": str
                The current status of the API (e.g., "ok").
            "version": str
                The version number of the API.
            "name": str
                The name of the API service.
            "description": str
                A brief description of the API's purpose.
        }

    Notes
    -----
    This endpoint is useful for:
    - Verifying that the API is up and running
    - Checking the current version of the API
    - Getting a quick overview of the API's purpose

    The response is intentionally lightweight to ensure fast response times,
    making it suitable for frequent health checks or monitoring.

    Examples
    --------
    >>> import httpx
    >>> response = httpx.get("http://api-url/")
    >>> print(response.json())
    {
        "status": "ok",
        "version": "0.1.0",
        "name": "DuckDB VSS API",
        "description": "Vector Similarity Search API powered by DuckDB"
    }
    """

    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "name": settings.APP_NAME,
        "description": "Vector Similarity Search API powered by DuckDB",
    }


@app.post("/s", description="Abbreviation for /search")
@app.post("/search", description="Perform a vector similarity search on the database")
async def api_search(
    response: Response,
    debug: bool = Query(default=False),
    request: SearchRequest = Body(
        ...,
        description="The search request containing the query and search parameters",
        openapi_examples={
            "search_request_example_1": {
                "summary": "Search Request Example 1",
                "value": {
                    "query": "How is Amazon?",
                    "top_k": 5,
                    "with_embedding": False,
                },
            },
            "search_request_example_2": {
                "summary": "Search Request Example 2: Base64",
                "value": {
                    "query": "",
                    "top_k": 5,
                    "with_embedding": False,
                },
            },
        },
    ),
    conn: duckdb.DuckDBPyConnection = Depends(
        lambda: duckdb.connect(settings.DUCKDB_PATH),
    ),
    time_start: float = Depends(lambda: time.perf_counter()),
) -> SearchResponse:
    """
    Perform a vector similarity search on the database.

    This endpoint processes a single search request, converting the input query
    into a vector embedding (if necessary) and performing a similarity search
    against the vector database.

    Parameters
    ----------
    response : Response
        The FastAPI response object, used to set custom headers.
    debug : bool, optional
        If True, prints debug information such as elapsed time (default is False).
    request : SearchRequest
        The search request object containing the query and search parameters.
    conn : duckdb.DuckDBPyConnection
        A connection to the DuckDB database.
    time_start : float
        The start time of the request processing, used for performance measurement.

    Returns
    -------
    SearchResponse
        An object containing the search results, including matched points,
        associated documents, and relevance scores.

    Raises
    ------
    HTTPException
        If there's an error processing the request or performing the search.

    Notes
    -----
    - The function first ensures that the input query is converted to a vector embedding.
    - It then performs a vector similarity search using the DuckDB database.
    - The search results are ordered by relevance (cosine similarity).
    - Performance metrics are added to the response headers.

    Examples
    --------
    >>> import httpx
    >>> response = httpx.post("http://api-url/search", json={
    ...     "query": "How does AI work?",
    ...     "top_k": 5,
    ...     "with_embedding": False
    ... })
    >>> print(response.json())
    {
        "results": [
            {
                "point": {...},
                "document": {...},
                "relevance_score": 0.95
            },
            ...
        ]
    }

    See Also
    --------
    SearchRequest : The model defining the structure of the search request.
    SearchResponse : The model defining the structure of the search response.
    vector_search : The underlying function performing the vector similarity search.
    """  # noqa: E501

    # Ensure vectors
    vectors = await SearchRequest.to_vectors(
        [request],
        cache=app.state.cache,
        openai_client=app.state.openai_client,
    )
    vector = vectors[0]

    # Search
    search_results = await VSS.vector_search(
        vector,
        top_k=request.top_k,
        embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
        documents_table_name=settings.DOCUMENTS_TABLE_NAME,
        points_table_name=settings.POINTS_TABLE_NAME,
        conn=conn,
        with_embedding=request.with_embedding,
    )

    # Return results
    time_end = time.perf_counter()
    elapsed_time_ms_str = f"{(time_end - time_start) * 1000:.2f}ms"
    response.headers["X-Processing-Time"] = elapsed_time_ms_str
    return SearchResponse.from_search_results(search_results)


@app.post("/bs", description="Abbreviation for /bulk_search")
@app.post(
    "/bulk_search",
    description="Perform multiple vector similarity searches in a single request",
)
async def api_bulk_search(
    response: Response,
    debug: bool = Query(default=False),
    request: BulkSearchRequest = Body(
        ...,
        description="The bulk search request containing multiple queries and search parameters",  # noqa: E501
        openapi_examples={
            "search_request_example_1": {
                "summary": "Bulk Search Request Example 1",
                "value": {
                    "queries": [
                        {
                            "query": "How is Apple doing?",
                            "top_k": 2,
                            "with_embedding": False,
                        },
                        {
                            "query": "What is the game score?",
                            "top_k": 2,
                            "with_embedding": False,
                        },
                    ],
                },
            },
        },
    ),
    time_start: float = Depends(lambda: time.perf_counter()),
) -> BulkSearchResponse:
    """
    Perform multiple vector similarity searches on the database in a single request.

    This endpoint processes a bulk search request, converting multiple input queries
    into vector embeddings (if necessary) and performing parallel similarity searches
    against the vector database.

    Parameters
    ----------
    response : Response
        The FastAPI response object, used to set custom headers.
    debug : bool, optional
        If True, prints debug information such as elapsed time (default is False).
    request : BulkSearchRequest
        The bulk search request object containing multiple queries and search parameters.
    time_start : float
        The start time of the request processing, used for performance measurement.

    Returns
    -------
    BulkSearchResponse
        An object containing the search results for all queries, including matched points,
        associated documents, and relevance scores for each query.

    Raises
    ------
    HTTPException
        If there's an error processing the request, such as no queries provided.

    Notes
    -----
    - The function first ensures that all input queries are converted to vector embeddings.
    - It then performs parallel vector similarity searches using the DuckDB database.
    - The search results for each query are ordered by relevance (cosine similarity).
    - Performance metrics are added to the response headers.
    - This bulk search is more efficient than making multiple individual search requests.

    Examples
    --------
    >>> import httpx
    >>> response = httpx.post("http://api-url/bulk_search", json={
    ...     "queries": [
    ...         {"query": "How does AI work?", "top_k": 3, "with_embedding": False},
    ...         {"query": "What is machine learning?", "top_k": 2, "with_embedding": True}
    ...     ]
    ... })
    >>> print(response.json())
    {
        "results": [
            {
                "results": [
                    {"point": {...}, "document": {...}, "relevance_score": 0.95},
                    {"point": {...}, "document": {...}, "relevance_score": 0.85},
                    {"point": {...}, "document": {...}, "relevance_score": 0.75}
                ]
            },
            {
                "results": [
                    {"point": {...}, "document": {...}, "relevance_score": 0.92},
                    {"point": {...}, "document": {...}, "relevance_score": 0.88}
                ]
            }
        ]
    }

    See Also
    --------
    BulkSearchRequest : The model defining the structure of the bulk search request.
    BulkSearchResponse : The model defining the structure of the bulk search response.
    vector_search : The underlying function performing individual vector similarity searches.
    ensure_vectors : Function to prepare input vectors for search.

    Notes
    -----
    The bulk search process can be visualized as follows:
    """  # noqa: E501

    if not request.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No queries provided.",
        )

    # Ensure vectors
    vectors = await SearchRequest.to_vectors(
        request.queries,
        cache=app.state.cache,
        openai_client=app.state.openai_client,
    )

    # Search
    bulk_search_results = await asyncio.gather(
        *[
            VSS.vector_search(
                vector,
                top_k=req_query.top_k,
                embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
                documents_table_name=settings.DOCUMENTS_TABLE_NAME,
                points_table_name=settings.POINTS_TABLE_NAME,
                conn=duckdb.connect(settings.DUCKDB_PATH),
                with_embedding=req_query.with_embedding,
            )
            for vector, req_query in zip(vectors, request.queries)
        ]
    )

    # Return results
    time_end = time.perf_counter()
    elapsed_time_ms_str = f"{(time_end - time_start) * 1000:.2f}ms"
    response.headers["X-Processing-Time"] = elapsed_time_ms_str
    return BulkSearchResponse.from_bulk_search_results(bulk_search_results)
