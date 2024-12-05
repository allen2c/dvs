import pathlib
import typing

import pytest
from fastapi.testclient import TestClient

import dvs.app
from dvs import DVS
from dvs.config import settings
from dvs.types.document import Document


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
