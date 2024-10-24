import asyncio
import base64
import binascii
import json
import pathlib
import time
from textwrap import dedent
from typing import Any, Dict, List, Literal, Optional, Text, Tuple, Union

import duckdb
import numpy as np
import openai
from diskcache import Cache
from fastapi import Body, Depends, FastAPI, HTTPException, Query, Response, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

column_names_with_embedding = ("point_id", "document_id", "content_md5", "embedding")
column_names_without_embedding = ("point_id", "document_id", "content_md5")
sql_stmt_vss = dedent(
    """
    WITH vector_search AS (
        SELECT {column_names_expr}, array_cosine_similarity(embedding, ?::FLOAT[{embedding_dimensions}]) AS relevance_score
        FROM {points_table_name}
        ORDER BY relevance_score DESC
        LIMIT {top_k}
    )
    SELECT p.*, d.*
    FROM vector_search p
    JOIN {documents_table_name} d ON p.document_id = d.document_id
    ORDER BY p.relevance_score DESC
    """  # noqa: E501
).strip()


async def to_vectors_with_cache(
    queries: Union[List[Text], Text], *, cache: Cache, openai_client: "openai.OpenAI"
) -> List[List[float]]:
    """"""

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
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
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


def decode_base64_to_vector(base64_str: Text) -> Optional[List[float]]:
    try:
        return np.frombuffer(  # type: ignore[no-untyped-call]
            base64.b64decode(base64_str), dtype="float32"
        ).tolist()
    except (binascii.Error, ValueError):
        return None  # not a base64 encoded string


async def ensure_vectors(
    queries: Union[List[Text], Text, List[List[float]], List[Union[Text, List[float]]]],
    *,
    cache: Cache,
    openai_client: "openai.OpenAI",
) -> List[List[float]]:
    queries = [queries] if isinstance(queries, Text) else queries

    output_vectors: List[Optional[List[float]]] = [None] * len(queries)
    required_emb_indices: List[int] = []
    required_emb_text: List[Text] = []
    for idx, query in enumerate(queries):
        query = query.strip() if isinstance(query, Text) else query
        if not query:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid queries[{idx}].",
            )
        if isinstance(query, Text):
            might_be_vector = decode_base64_to_vector(query)
            if might_be_vector is None:
                required_emb_indices.append(idx)
                required_emb_text.append(query)
            else:
                output_vectors[idx] = might_be_vector
        else:
            output_vectors[idx] = query

    # Ensure all required embeddings are text
    assert len(required_emb_indices) == len(required_emb_text)
    if required_emb_indices:
        embeddings = await to_vectors_with_cache(
            required_emb_text, cache=cache, openai_client=openai_client
        )
        for idx, embedding in zip(required_emb_indices, embeddings):
            output_vectors[idx] = embedding

    # Ensure all vectors are not None
    assert all(v is not None for v in output_vectors)
    return output_vectors  # type: ignore


async def vector_search(
    vector: List[float],
    *,
    top_k: int,
    embedding_dimensions: int,
    documents_table_name: Text,
    points_table_name: Text,
    conn: "duckdb.DuckDBPyConnection",
    with_embedding: bool = True,
) -> List[Tuple["Point", Optional["Document"], float]]:
    output: List[Tuple["Point", Optional["Document"], float]] = []

    column_names_expr = ", ".join(
        list(
            column_names_with_embedding
            if with_embedding
            else column_names_without_embedding
        )
    )
    query = sql_stmt_vss.format(
        top_k=top_k,
        column_names_expr=column_names_expr,
        embedding_dimensions=embedding_dimensions,
        documents_table_name=documents_table_name,
        points_table_name=points_table_name,
    )
    params = [vector]

    # Fetch results
    result = await asyncio.to_thread(conn.execute, query, params)
    fetchall_result = await asyncio.to_thread(result.fetchall)

    # Convert to output format
    assert result.description is not None
    for row in fetchall_result:
        row_dict = dict(zip([desc[0] for desc in result.description], row))
        row_dict["metadata"] = json.loads(row_dict.get("metadata") or "{}")
        row_dict["embedding"] = row_dict.get("embedding") or []
        output.append(
            (
                Point.model_validate(row_dict),
                Document.model_validate(row_dict),
                row_dict["relevance_score"],
            )
        )

    return output


class Settings(BaseSettings):
    APP_NAME: Text = Field(default="duckdb-vss-api")
    APP_VERSION: Text = Field(default="0.1.0")
    APP_ENV: Literal["development", "production", "test"] = Field(default="development")

    # DuckDB
    DUCKDB_PATH: Text = Field(default="./documents.duckdb")
    POINTS_TABLE_NAME: Text = Field(default="points")
    DOCUMENTS_TABLE_NAME: Text = Field(default="documents")
    EMBEDDING_MODEL: Text = Field(default="text-embedding-3-small")
    EMBEDDING_DIMENSIONS: int = Field(default=512)

    # OpenAI
    OPENAI_API_KEY: Optional[Text] = None

    # Embeddings
    CACHE_PATH: Text = Field(default="./embeddings.cache")
    CACHE_SIZE_LIMIT: int = Field(default=100 * 2**20)  # 100MB

    def validate_variables(self):
        if not pathlib.Path(self.DUCKDB_PATH).exists():
            raise ValueError(f"Database file does not exist: {self.DUCKDB_PATH}")
        else:
            self.DUCKDB_PATH = str(pathlib.Path(self.DUCKDB_PATH).resolve())


class Point(BaseModel):
    point_id: Text
    document_id: Text
    content_md5: Text
    embedding: List[float]


class Document(BaseModel):
    document_id: Text
    name: Text
    content: Text
    content_md5: Text
    metadata: Optional[Dict[Text, Any]] = Field(default_factory=dict)
    created_at: Optional[int] = Field(default=None)
    updated_at: Optional[int] = Field(default=None)


class SearchRequest(BaseModel):
    query: Text | List[float]
    top_k: int = Field(default=5)
    with_embedding: bool = Field(default=False)


class BulkSearchRequest(BaseModel):
    queries: List[SearchRequest]


class SearchResult(BaseModel):
    point: Point
    document: Optional[Document] = Field(default=None)
    relevance_score: float = Field(default=0.0)

    @classmethod
    def from_search_result(
        cls, search_result: Tuple["Point", Optional["Document"], float]
    ) -> "SearchResult":
        return cls.model_validate(
            {
                "point": search_result[0],
                "document": search_result[1],
                "relevance_score": search_result[2],
            }
        )


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)

    @classmethod
    def from_search_results(
        cls,
        search_results: List[Tuple["Point", Optional["Document"], float]],
    ) -> "SearchResponse":
        out = cls.model_validate(
            {
                "results": [
                    SearchResult.from_search_result(res) for res in search_results
                ]
            }
        )
        return out


class BulkSearchResponse(BaseModel):
    results: List[SearchResponse] = Field(default_factory=list)

    @classmethod
    def from_bulk_search_results(
        cls,
        bulk_search_results: List[List[Tuple["Point", Optional["Document"], float]]],
    ) -> "BulkSearchResponse":
        return cls.model_validate(
            {
                "results": [
                    SearchResponse.from_search_results(search_results)
                    for search_results in bulk_search_results
                ]
            }
        )


settings = Settings()


app = FastAPI()
app.state.settings = app.extra["settings"] = settings
app.state.cache = app.extra["cache"] = Cache(
    directory=settings.CACHE_PATH, size_limit=settings.CACHE_SIZE_LIMIT
)
with duckdb.connect(settings.DUCKDB_PATH) as __conn__:
    __conn__.sql("INSTALL json;")
    __conn__.sql("LOAD json;")
    __conn__.sql("INSTALL vss;")
    __conn__.sql("LOAD vss;")
if settings.OPENAI_API_KEY is None:
    app.state.openai_client = app.extra["openai_client"] = None
else:
    app.state.openai_client = app.extra["openai_client"] = openai.OpenAI(
        api_key=settings.OPENAI_API_KEY
    )


@app.get("/")
async def api_root():
    return {"status": "ok"}


@app.post("/s")
@app.post("/search")
async def api_search(
    response: Response,
    debug: bool = Query(default=False),
    request: SearchRequest = Body(
        ...,
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
                    "query": "XQHFva81cb2hV3Q88ZahPZ7LzzyqKWy9K9tovSSJ0T3wUI+83+yPuS++wjy0b4A7mlySvRWObb3OjrI6nug2PYEsvbz9CwM9VYaCve8QHzwm7Mq8nGhWPbzqwrxRhsE8VEy0Pb1HGj3PyIC94PjTukTuVj0dGtO9GVRgPZiFfL2mgOA8MoQ1vJoiRD0BtY49eD3eu9DxqzypQDE97u0VPVBMc71EBZw8W5gpPV9kvrzlG568uUE3vc+adr2bf5u9AF7ZPB+as7uO4Y68Wlg5PK9SWD1dGIq8AHvAvCvb6LxqtpY9dKsXPXhaRb0iCXE8JQOQPAG1jrwr+M88w1Ofu0K/Cb1OFwS93w+ZvRbIOz2jCwE9qToPvYnVCbwM8CE9ELaUu6F0WzzQ1EQ9lnMWPT85hzw/OQe9yqtYPIJVaLzTdy69SO4XPAqHhjw0yke99DmLPeFykj1s38G6oYsgvVMjCbzU2qe9vAGIO7geLj0/HCC9mKJjPbbYmzw51kw80RrXPAB1Hj2Nitm8K/jPPLb1AryqKey8a6XzO8wO0rsmQ4A9QFyQvUy6LD1zTsC9yqvYPMb2CD0Zcce9UGlavACSBb30HCS9FJ+QvJUzJryMYa68xwLNvACYJzkwG5q9oXTbPAIYCLwTK3S9qADBOTTnLr3H5WW8jaEePZPQrL1UL808pFGTvZ4ihT0tPmI7Eyv0O258CT3TYOk8VWmbvFBMczumY/k8UGlaO5eWH71tJVS9G7dZuq3Sd7v++l+93BX6PGcNizlmzZq8MT4jPPPiVb1s/Ci9a6XzPFuBZDz3bnq9CCQNPd6VWryNvgU9iuFNPb5NPD12FDM9YyqxPJGKmr2hiyA9nIU9vaujqruQJyG85QTZPYUPF72Djza8ttibvT1/WLlHrie9t8FWPRCfTz2+ZAE9xvYIvWlf4bysqcw8UuMYvTmcfjwprJu9cjFZvbgkUDqvUli96iejvRC2lL2RM2W8cLF4vaRREzwBuzC9DLz1vCWyfL3xYvU910PDPWlfYT3L5Sa7z+sJPQVN97wtW0m9rKlMPIxEx7zIK/g8kTNlvHg9Xj2a6PW8ELYUva818bzpBJo9YQEGPmIkD71iJA+9m3+bvDJKZ73V/TA9j9BrPbyw9DwOH++8X2S+vRlxRz0YK7U8aPxnvKV0nDp5t5w9VsyUvPGWIT0kbOq88DnKPcqO8Tvjod+8AzuRPPPFbr2nnUc8b6U0vazAETtXu3G9ossQvUQLPj07OUa9fWAoPRXCmToZN3m9PX/YPG9r5jz8y5I8Uq/svDVEBrv7qAm6Jc/jPLP7Y70B2Je8Rm43PJBKqr3wOcq7T10WvSD3ir2DyQS9ru/evNsmnbw8cxQ9Dh/vPeFykr0//7i9k7PFNE4dJrx9YKi9g482vWCkrjx+oBg8ql0YvJ7Lz7ysjOW852dSvWaZbrzxYnU8jwQYvW58Cb3FvLq9apmvvM+a9rzqEN69PzkHPfnRc70NMJI7ifgSPcsCDj3zFgK8tG8AvXdUozxJ3XS7tbUSvF3H9jsbt9m8DnCCOm5fojzDcIY8QAt9vZ8uSb0Fh0U8WTvSu6VXNTx4WsU8j9DrPYSb+jxMuiy9J0kiuh1xiL3XJtw9EfwmvOKygjwgw169n0UOvdXgSb3y1pE8q50IPh59zLuCVWi9pFGTvQqqD7y/qhO9TyNIvK9S2L3Fn9M8NOcuvS/bqTzWPSG8V7vxvDm5Zbychb28I30NvdEa170KjSi9yqvYPCsyHjwEXpq7Kw8VvYS4Yb3l53G98tYRvXHOX7wrFTe7JMOfvLP7Y738yxI9FuWivNdgKr1AKOQ8mRaAvfy0zbto/Oc8uV6eO/T/PD0nT0Q8hjhCvU3dNT38ev+8T0AvPZX5V70kw588EuVhPLfePb0TXyA9I6AWvSImWLyhrik9Gs4ePZZWLz128ak9anxIvY/tUjumgGA8sdI4PeiQ/TxhqlA9k7NFvJ7oNrzVw+I6IiZYvQG1Dj1QTHM93skGPWRT3DwUZcI82s/nPHDfgrs2LUE8l3k4unSU0jxnDQs7NfNyPYobHL36NO08ru9ePWa2Vb2z+2M9W4FkvIEP1rwHzdc6bSVUPLWSiTxOOo28zCWXu0TRbz0nMl29niIFPPVitr1kcMM8KuyLvdFOA71TBqK8t8FWvVp1ILzHH7Q5wOqDvbCpjT1/ycM76eeyvInVib1ZO9K8cc5fPHCx+LrjuKQ9RWgVvGzfwTzIfAu9N20xvQfqvjvM8Wo8nug2vG5for3xliE92aY8vT2cPz0azh49eJSTvcb2CDxtHzK903euPeJ4NLzKq9i8X0fXuxlUYD2kLgo9ql2Yu9Ea17s+xWq9vAcqvfnuWrzSVCU71wl1PdcmXD0/OYe9b4jNu7pNe71ARUu9uoGnPR5D/rxr2R874CyAPdrPZ73gLIC9PsVqvXm3nDyIYe28W5ipPMW8ujyj9Du7P/+4u75NvLwpyYI9evcMvQqqjzxljaq99pyEPALBUj2jESM9b8IbOnEIrjyRFv68VpLGvJPtEzw2Zw+8VYaCOyTghj0jg6+8isTmO1avLb1WdV89lTOmPH/DIT0OWb08LqHbPdWm+7tySB69pR3nPcwO0ryzL5C7ZGqhPKgAwbpZO9K9Th0mvS14ML1oNja5Fsg7PYSber3eePO8CqqPvcflZbxI0bA8er0+PeJ4tLzzxW69Q8UrPSwh+zw=",  # noqa: E501
                    "top_k": 5,
                    "with_embedding": False,
                },
            },
        },
    ),
    conn: duckdb.DuckDBPyConnection = Depends(
        lambda: duckdb.connect(settings.DUCKDB_PATH)
    ),
    time_start: float = Depends(lambda: time.perf_counter()),
):
    # Ensure vectors
    vectors = await ensure_vectors(
        [request.query],
        cache=app.state.cache,
        openai_client=app.state.openai_client,
    )
    vector = vectors[0]

    # Search
    search_results = await vector_search(
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
    if debug:
        print(f"Elapsed time: {elapsed_time_ms_str}")
    return SearchResponse.from_search_results(search_results)


@app.post("/bs")
@app.post("/bulk_search")
async def api_bulk_search(
    response: Response,
    debug: bool = Query(default=False),
    request: BulkSearchRequest = Body(
        ...,
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
):
    if not request.queries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No queries provided.",
        )

    # Ensure vectors
    vectors = await ensure_vectors(
        [query.query for query in request.queries],
        cache=app.state.cache,
        openai_client=app.state.openai_client,
    )

    # Search
    bulk_search_results = await asyncio.gather(
        *[
            vector_search(
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
    if debug:
        print(f"Elapsed time: {elapsed_time_ms_str}")
    return BulkSearchResponse.from_bulk_search_results(bulk_search_results)
