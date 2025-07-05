import pytest

import dvs
from dvs.types.document import Document
from dvs.types.point import Point


def test_dvs_db_touch(dvs_client: dvs.DVS):
    assert dvs_client.db.touch()


def test_dvs_add(dvs_client: dvs.DVS):
    """Test adding documents to the database."""
    # Test data
    test_documents = [
        "This is the first test document about machine learning.",
        "This is the second test document about artificial intelligence.",
        "This is the third test document about data science.",
    ]

    # Add documents
    result = dvs_client.add(test_documents)

    # Verify the result
    assert result["success"] is True
    assert result["created_documents"] == 3
    assert result["ignored_documents"] == 0
    assert result["created_points"] == 3
    assert result["error"] is None

    # Test adding the same documents again (should be ignored)
    result_duplicate = dvs_client.add(test_documents)
    assert result_duplicate["success"] is True
    assert result_duplicate["created_documents"] == 0
    assert result_duplicate["ignored_documents"] == 3


def test_dvs_remove(dvs_client: dvs.DVS):
    """Test removing documents from the database."""
    # First add a document
    test_document = "This is a test document that will be removed."
    dvs_client.add(test_document)

    # Get the document ID from the database
    docs_pagination = dvs_client.db.documents.list()
    doc_id = docs_pagination.data[0].document_id

    # Remove the document
    dvs_client.remove(doc_id)

    # Verify the document is removed
    remaining_docs = dvs_client.db.documents.list()
    doc_ids = [doc.document_id for doc in remaining_docs.data]
    assert doc_id not in doc_ids


@pytest.mark.asyncio
async def test_dvs_search(dvs_client: dvs.DVS):
    """Test searching for documents in the database."""
    # Add test documents
    test_documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Deep learning uses neural networks with multiple layers.",
    ]

    dvs_client.add(test_documents)

    # Search for documents
    query = "artificial intelligence"
    results = await dvs_client.search(query, top_k=2)

    # Verify search results
    assert len(results) <= 2
    assert len(results) > 0

    # Each result should be a tuple of (Point, Document, score)
    for result in results:
        assert len(result) == 3
        point, document, score = result
        assert isinstance(point, Point)
        assert isinstance(document, Document)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Cosine similarity should be between 0 and 1
