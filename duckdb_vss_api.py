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
from fastapi import Body, Depends, FastAPI, HTTPException, Query, status
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
            await asyncio.to_thread(cache.set, queries[idx], embedding.embedding)
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
    queries: List[Union[Text, List[float]]]
    top_k: int = Field(default=5)
    with_embedding: bool = Field(default=False)


class SearchResult(BaseModel):
    point: Point
    document: Optional[Document] = Field(default=None)
    relevance_score: float = Field(default=0.0)


class SearchResponse(BaseModel):
    results: List[SearchResult] = Field(default_factory=list)
    elapsed_time_ms: float = Field(default=0.0)


class BulkSearchResponse(BaseModel):
    results: List[List[SearchResult]] = Field(default_factory=list)
    elapsed_time_ms: float = Field(default=0.0)


settings = Settings()


app = FastAPI()
app.state.settings = app.extra["settings"] = settings
app.state.cache = app.extra["cache"] = Cache(
    directory=settings.CACHE_PATH, size_limit=settings.CACHE_SIZE_LIMIT
)
with duckdb.connect(settings.DUCKDB_PATH) as conn:
    conn.sql("INSTALL json;")
    conn.sql("LOAD json;")
    conn.sql("INSTALL vss;")
    conn.sql("LOAD vss;")
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
    elapsed_time_ms = (time_end - time_start) * 1000
    if debug:
        print(f"Elapsed time: {elapsed_time_ms:.2f} ms")
    return SearchResponse.model_validate(
        {
            "results": [
                {
                    "point": res[0],
                    "document": res[1],
                    "relevance_score": res[2],
                }
                for res in search_results
            ],
            "elapsed_time_ms": elapsed_time_ms,
        }
    )
