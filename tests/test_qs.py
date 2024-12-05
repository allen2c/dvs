import pathlib
import typing

import openai
import pytest

from dvs.config import settings
from dvs.types.document import Document


def test_documents_qs(documents: typing.List[Document]):
    pathlib.Path(settings.DUCKDB_PATH).unlink(missing_ok=True)  # clean up

    # Touch
    Document.objects.touch(conn=settings.duckdb_conn, debug=True)

    # Create and bulk create
    _doc = Document.objects.create(documents[0], conn=settings.duckdb_conn, debug=True)
    Document.objects.bulk_create(documents[1:], conn=settings.duckdb_conn, debug=True)

    # Query
    _retrieved_doc = Document.objects.retrieve(
        _doc.document_id, conn=settings.duckdb_conn, debug=True
    )
    assert _retrieved_doc is not None
    assert _retrieved_doc.document_id == _doc.document_id

    # List
    _list_docs = Document.objects.list(conn=settings.duckdb_conn, debug=True)
    assert len(_list_docs.data) > 0

    # Gen
    _gen_docs = [
        doc for doc in Document.objects.gen(conn=settings.duckdb_conn, debug=True)
    ]
    assert len(_gen_docs) == len(documents)

    # Count
    _count = Document.objects.count(conn=settings.duckdb_conn, debug=True)
    assert _count == len(documents)

    # Delete
    _delete_doc_id = _doc.document_id
    Document.objects.remove(
        document_id=_delete_doc_id, conn=settings.duckdb_conn, debug=True
    )
    with pytest.raises(openai.NotFoundError):
        Document.objects.retrieve(
            document_id=_delete_doc_id,
            conn=settings.duckdb_conn,
            debug=True,
        )
    assert (
        Document.objects.count(
            document_id=_delete_doc_id,
            conn=settings.duckdb_conn,
            debug=True,
        )
        == 0
    )
