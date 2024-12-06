import pathlib
import typing

import pytest
from fastapi.testclient import TestClient

import dvs.app
import dvs.utils.to as TO
from dvs import DVS
from dvs.config import console, fake, settings
from dvs.types.document import Document
from dvs.types.search_response import SearchResponse

test_vector = (
    settings.openai_client.embeddings.create(
        input="test",
        model=settings.EMBEDDING_MODEL,
        dimensions=settings.EMBEDDING_DIMENSIONS,
    )
    .data[0]
    .embedding
)


@pytest.fixture(scope="module")
def fastapi_client(documents: typing.List[Document]):
    pathlib.Path(settings.DUCKDB_PATH).unlink(missing_ok=True)  # clean up

    # Initialize DVS
    client = DVS()
    client.add(documents, debug=True)

    with TestClient(dvs.app.app) as client:
        yield client


def test_health(fastapi_client: TestClient):
    response = fastapi_client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    response = fastapi_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize(
    "search_request",
    [
        {"query": "test"},
        {"query": fake.paragraph()},
        {"query": "test", "encoding": "plaintext"},
        {"query": "test", "top_k": 10},
        {"query": "test", "with_embedding": True},
        {"query": test_vector},
        {"query": test_vector, "encoding": "vector"},
        {"query": TO.vector_to_base64(test_vector)},
        {"query": TO.vector_to_base64(test_vector), "encoding": "base64"},
    ],
)
def test_search(fastapi_client: TestClient, search_request: typing.Dict):
    response = fastapi_client.post("/search", json=search_request)
    if response.status_code >= 300:
        console.print(f"[{response.status_code}] {response.text}", style="red")
    response.raise_for_status()
    search_response = SearchResponse.model_validate(response.json())
    assert len(search_response.results) > 0
