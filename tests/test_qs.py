import pathlib
import typing

import openai
import pytest

from dvs.config import settings
from dvs.types.document import Document
from dvs.types.point import Point


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


def test_points_qs(documents: typing.List[Document]):
    pathlib.Path(settings.DUCKDB_PATH).unlink(missing_ok=True)  # clean up

    # Touch
    Document.objects.touch(conn=settings.duckdb_conn, debug=True)
    Point.objects.touch(conn=settings.duckdb_conn, debug=True)

    # Points with embeddings from documents
    _points = [pt for doc in documents for pt in doc.to_points()]
    _points = Point.set_embeddings_from_contents(_points, documents, debug=True)

    # Create documents
    Document.objects.bulk_create(documents, conn=settings.duckdb_conn, debug=True)

    # Create
    _point = Point.objects.create(_points[0], conn=settings.duckdb_conn, debug=True)

    # Bulk create
    Point.objects.bulk_create(_points[1:], conn=settings.duckdb_conn, debug=True)

    # Retrieve
    _retrieved_point = Point.objects.retrieve(
        _point.point_id, conn=settings.duckdb_conn, debug=True
    )
    assert _retrieved_point is not None
    assert _retrieved_point.point_id == _point.point_id

    # List
    _list_points = Point.objects.list(conn=settings.duckdb_conn, debug=True)
    assert len(_list_points.data) > 0
    _list_points = Point.objects.list(
        document_id=_point.document_id, conn=settings.duckdb_conn, debug=True
    )
    assert len(_list_points.data) > 0
    assert all(pt.document_id == _point.document_id for pt in _list_points.data)

    # Gen
    _gen_points = [
        pt for pt in Point.objects.gen(conn=settings.duckdb_conn, debug=True)
    ]
    assert len(_gen_points) == len(_points)

    # Count
    _count = Point.objects.count(conn=settings.duckdb_conn, debug=True)
    assert _count == len(_points)

    # Delete
    _delete_point_id = _point.point_id
    _delete_point_doc_id = _point.document_id
    Point.objects.remove(_delete_point_id, conn=settings.duckdb_conn, debug=True)
    with pytest.raises(openai.NotFoundError):
        Point.objects.retrieve(
            point_id=_delete_point_id, conn=settings.duckdb_conn, debug=True
        )
    assert (
        Point.objects.count(
            document_id=_delete_point_doc_id, conn=settings.duckdb_conn, debug=True
        )
        == 0
    )
