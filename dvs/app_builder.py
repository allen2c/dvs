import asyncio
import time
from textwrap import dedent
from typing import Dict, Text

import duckdb
import openai
from diskcache import Cache
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Response, status

import dvs.utils.vss as VSS
from dvs.config import settings
from dvs.types.bulk_search_request import BulkSearchRequest
from dvs.types.bulk_search_response import BulkSearchResponse
from dvs.types.search_request import SearchRequest
from dvs.types.search_response import SearchResponse


def init_app() -> FastAPI:
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
    return app


def build_app_state(app: FastAPI) -> FastAPI:
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

    return app


def build_app_resources(app: FastAPI) -> FastAPI:
    # DuckDB connection
    with duckdb.connect(settings.DUCKDB_PATH) as __conn__:
        __conn__.sql("INSTALL json;")
        __conn__.sql("LOAD json;")
        __conn__.sql("INSTALL vss;")
        __conn__.sql("LOAD vss;")

    return app


def build_app() -> FastAPI:
    # FastAPI app
    app = init_app()
    # FastAPI app state
    app = build_app_state(app)
    # FastAPI app resources
    app = build_app_resources(app)

    # API endpoints
    @app.get(
        "/", description="Root endpoint providing API status and basic information"
    )
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
    @app.post(
        "/search", description="Perform a vector similarity search on the database"
    )
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
        t0_api: float = Depends(lambda: time.perf_counter()),
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
        t0_api : float
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
        t0_vec = time.perf_counter()
        vectors = await SearchRequest.to_vectors(
            [request],
            cache=app.state.cache,
            openai_client=app.state.openai_client,
        )
        vector = vectors[0]
        t1_vec = time.perf_counter()

        # Search
        t0_db = time.perf_counter()
        search_results = await VSS.vector_search(
            vector,
            top_k=request.top_k,
            embedding_dimensions=settings.EMBEDDING_DIMENSIONS,
            documents_table_name=settings.DOCUMENTS_TABLE_NAME,
            points_table_name=settings.POINTS_TABLE_NAME,
            conn=conn,
            with_embedding=request.with_embedding,
        )
        t1_db = time.perf_counter()

        # Return results
        response.headers["Server-Timing"] = (
            f"total;dur={(time.perf_counter() - t0_api) * 1000:.2f},"
            f"vec;dur={(t1_vec - t0_vec) * 1000:.2f},"
            f"db;dur={(t1_db - t0_db) * 1000:.2f}"
        )
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
        t0_api: float = Depends(lambda: time.perf_counter()),
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
        t0_api : float
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
        t0_vec = time.perf_counter()
        vectors = await SearchRequest.to_vectors(
            request.queries,
            cache=app.state.cache,
            openai_client=app.state.openai_client,
        )
        t1_vec = time.perf_counter()

        # Search
        t0_db = time.perf_counter()
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
        t1_db = time.perf_counter()

        # Return results
        t1_api = time.perf_counter()
        response.headers["Server-Timing"] = (
            f"total;dur={(t1_api - t0_api) * 1000:.2f},"
            f"vec;dur={(t1_vec - t0_vec) * 1000:.2f},"
            f"db;dur={(t1_db - t0_db) * 1000:.2f}"
        )
        return BulkSearchResponse.from_bulk_search_results(bulk_search_results)

    return app
