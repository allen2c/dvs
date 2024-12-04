import pathlib
from typing import Literal, Optional, Text

import diskcache
import openai
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings
from rich.console import Console

console = Console()


class Settings(BaseSettings):
    """
    Settings for the DuckDB VSS API.

    This class defines the configuration parameters for the DuckDB Vector Similarity Search (VSS) API.
    It uses Pydantic's BaseSettings for easy environment variable loading and validation.
    """  # noqa: E501

    APP_NAME: Text = Field(
        default="DVS",
        description="The name of the application. Used for identification and logging purposes.",  # noqa: E501
    )
    APP_VERSION: Text = Field(
        default="0.1.0",
        description="The version of the application. Follows semantic versioning.",
    )
    APP_ENV: Literal["development", "production", "test"] = Field(
        default="development",
        description="The environment in which the application is running. Affects logging and behavior.",  # noqa: E501
    )
    APP_READY: bool = Field(
        default=False,
        description="Whether the application is ready to serve requests.",
    )

    # DuckDB
    DUCKDB_PATH: Text = Field(
        default="./data/documents.duckdb",
        description="The file path to the DuckDB database file containing document and embedding data.",  # noqa: E501
    )
    POINTS_TABLE_NAME: Text = Field(
        default="points",
        description="The name of the table in DuckDB that stores the vector embeddings and related point data.",  # noqa: E501
    )
    DOCUMENTS_TABLE_NAME: Text = Field(
        default="documents",
        description="The name of the table in DuckDB that stores the document metadata.",  # noqa: E501
    )
    EMBEDDING_MODEL: Text = Field(
        default="text-embedding-3-small",
        description="The name of the OpenAI embedding model to use for generating vector embeddings.",  # noqa: E501
    )
    EMBEDDING_DIMENSIONS: int = Field(
        default=512,
        description="The number of dimensions in the vector embeddings generated by the chosen model.",  # noqa: E501
    )

    # OpenAI
    OPENAI_API_KEY: Optional[Text] = Field(
        default=None,
        description="The API key for authenticating with OpenAI services. If not provided, OpenAI features will be disabled.",  # noqa: E501
    )

    # Embeddings
    CACHE_PATH: Text = Field(
        default="./.cache/embeddings.cache",
        description="The file path to the cache directory for storing computed embeddings.",  # noqa: E501
    )
    CACHE_SIZE_LIMIT: int = Field(
        default=100 * 2**20,
        description="The maximum size of the embeddings cache in bytes. Default is 100MB.",  # noqa: E501
    )

    # Properties
    _openai_client: Optional[openai.OpenAI] = PrivateAttr(default=None)
    _cache: Optional[diskcache.Cache] = PrivateAttr(default=None)

    @property
    def openai_client(self) -> openai.OpenAI:
        if self._openai_client is None:
            self._openai_client = openai.OpenAI(api_key=self.OPENAI_API_KEY)
        return self._openai_client

    @property
    def cache(self) -> diskcache.Cache:
        if self._cache is None:
            self._cache = diskcache.Cache(
                self.CACHE_PATH, size_limit=self.CACHE_SIZE_LIMIT
            )
        return self._cache

    def validate_variables(self):
        """
        Validate the variables in the settings.
        """

        if not pathlib.Path(self.DUCKDB_PATH).exists():
            self.APP_READY = False
        else:
            self.DUCKDB_PATH = str(pathlib.Path(self.DUCKDB_PATH).resolve())
            self.APP_READY = True


settings = Settings()
settings.validate_variables()
