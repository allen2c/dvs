import pathlib
import time
from typing import Any, Dict, List, Literal, Text

from diskcache import Cache
from fastapi import FastAPI
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


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
    relevance_score: float


class Document(BaseModel):
    document_id: Text
    name: Text
    content: Text
    content_md5: Text
    metadata: Dict[Text, Any] = Field(default_factory=dict)
    created_at: int = Field(default_factory=lambda: int(time.time()))
    updated_at: int = Field(default_factory=lambda: int(time.time()))


class SearchRequest(BaseModel):
    query: Text | List[float]
    top_k: int = Field(default=5)
    with_embedding: bool = Field(default=False)
    with_documents: bool = Field(default=False)


settings = Settings()


app = FastAPI()
app.state.settings = app.extra["settings"] = settings
app.state.cache = app.extra["cache"] = Cache(
    directory=settings.CACHE_PATH, size_limit=settings.CACHE_SIZE_LIMIT
)


@app.get("/")
async def api_root():
    return {"status": "ok"}


@app.post("/s")
@app.post("/search")
async def api_search():
    return {"status": "ok"}
