import math
import os
import pathlib
import random
import tempfile
import typing

import pytest

if typing.TYPE_CHECKING:
    from dvs.types.document import Document


def pytest_configure(config):
    os.environ["APP_ENV"] = "test"
    os.environ["DUCKDB_PATH"] = str(pathlib.Path(tempfile.mkdtemp()) / "test.duckdb")


@pytest.fixture(scope="session")
def documents(ratio: float = 0.01) -> typing.List["Document"]:
    print("\nLoading fixture: 'documents'")
    from dvs.utils.datasets import download_documents

    docs = download_documents(name="bbc", overwrite=False)
    assert len(docs) > 0, "No documents found"
    docs = random.sample(docs, math.ceil(len(docs) * ratio))
    print(f"Loaded {len(docs)} sampled documents")
    return docs
