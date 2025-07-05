import os
import pathlib
import tempfile

import diskcache
import openai
import openai_embeddings_model as oai_emb_model
import pytest

import dvs

MODEL = "text-embedding-3-small"
DIMENSIONS = 512
EMBEDDING_CACHE_PATH = pathlib.Path("./cache/dvs/embeddings.cache")


def pytest_configure(config):
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DUCKDB_PATH"] = str(pathlib.Path(tempfile.mkdtemp()) / "test.duckdb")


@pytest.fixture(scope="module")
def temp_dir() -> pathlib.Path:
    return pathlib.Path(tempfile.mkdtemp()) / "test.duckdb"


@pytest.fixture(scope="module")
def cache() -> diskcache.Cache:
    return diskcache.Cache(EMBEDDING_CACHE_PATH)


@pytest.fixture(scope="module")
def openai_client() -> openai.OpenAI:
    return openai.OpenAI()


@pytest.fixture(scope="module")
def model(
    cache: diskcache.Cache, openai_client: openai.OpenAI
) -> oai_emb_model.OpenAIEmbeddingsModel:
    return oai_emb_model.OpenAIEmbeddingsModel(
        model=MODEL,
        openai_client=openai_client,
        cache=cache,
    )


@pytest.fixture(scope="module")
def model_settings() -> oai_emb_model.ModelSettings:
    return oai_emb_model.ModelSettings(
        dimensions=DIMENSIONS,
    )


@pytest.fixture(scope="module")
def dvs_settings() -> dvs.Settings:
    return dvs.Settings(DUCKDB_PATH=str(temp_dir()))


@pytest.fixture(scope="module")
def dvs_client(
    dvs_settings: dvs.Settings,
    model: oai_emb_model.OpenAIEmbeddingsModel,
    model_settings: oai_emb_model.ModelSettings,
) -> dvs.DVS:
    return dvs.DVS(
        dvs_settings, model=model, model_settings=model_settings, verbose=True
    )
