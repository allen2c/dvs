import pytest
from openai import NotFoundError

import dvs
from dvs.types.document import Document


def test_dvs_db_documents_touch(dvs_client: dvs.DVS):
    assert dvs_client.db.documents.touch()


def test_dvs_db_documents_retrieve(dvs_client: dvs.DVS):
    # Setup: Create a document to retrieve
    dvs_client.db.documents.touch()
    doc = Document.from_content(
        content="This is a test document for retrieval",
        name="Test Document",
        metadata={"category": "test"},
    )
    created_doc = dvs_client.db.documents.create(doc)

    # Test: Retrieve the document
    retrieved_doc = dvs_client.db.documents.retrieve(created_doc.document_id)

    # Assertions
    assert retrieved_doc.document_id == created_doc.document_id
    assert retrieved_doc.name == "Test Document"
    assert retrieved_doc.content == "This is a test document for retrieval"
    assert retrieved_doc.metadata == {"category": "test"}

    # Test: Retrieve non-existent document should raise NotFoundError
    with pytest.raises(NotFoundError):
        dvs_client.db.documents.retrieve("non-existent-id")


def test_dvs_db_documents_create(dvs_client: dvs.DVS):
    # Setup
    dvs_client.db.documents.touch()

    # Test: Create from Document object
    doc = Document.from_content(
        content="Hello, world! This is a test document.",
        name="My Test Document",
        metadata={"author": "test_user", "type": "sample"},
    )
    created_doc = dvs_client.db.documents.create(doc)

    # Assertions
    assert created_doc.document_id is not None
    assert created_doc.name == "My Test Document"
    assert created_doc.content == "Hello, world! This is a test document."
    assert created_doc.metadata == {"author": "test_user", "type": "sample"}
    assert created_doc.content_md5 is not None

    # Test: Create from dictionary
    doc_dict = {
        "name": "Dict Document",
        "content": "This document was created from a dictionary.",
        "content_md5": Document.hash_content(
            "This document was created from a dictionary."
        ),
        "metadata": {"source": "dict"},
    }
    created_from_dict = dvs_client.db.documents.create(doc_dict)

    # Assertions
    assert created_from_dict.document_id is not None
    assert created_from_dict.name == "Dict Document"
    assert created_from_dict.content == "This document was created from a dictionary."
    assert created_from_dict.metadata == {"source": "dict"}


def test_dvs_db_documents_bulk_create(dvs_client: dvs.DVS):
    # Setup
    dvs_client.db.documents.touch()

    # Test: Create multiple documents
    docs = []
    for i in range(3):
        doc = Document.from_content(
            content=f"This is test document number {i+1}",
            name=f"Document {i+1}",
            metadata={"index": i, "batch": "test"},
        )
        docs.append(doc)

    created_docs = dvs_client.db.documents.bulk_create(docs)

    # Assertions
    assert len(created_docs) == 3
    for i, doc in enumerate(created_docs):
        assert doc.document_id is not None
        assert doc.name == f"Document {i+1}"
        assert doc.content == f"This is test document number {i+1}"
        assert doc.metadata == {"index": i, "batch": "test"}

    # Test: Create from mixed types (Documents and dicts)
    mixed_docs = [
        Document.from_content("Content from Document object", name="Doc Object"),
        {
            "name": "Dict Document",
            "content": "Content from dictionary",
            "content_md5": Document.hash_content("Content from dictionary"),
            "metadata": {"type": "dict"},
        },
    ]

    created_mixed = dvs_client.db.documents.bulk_create(mixed_docs)
    assert len(created_mixed) == 2
    assert created_mixed[0].name == "Doc Object"
    assert created_mixed[1].name == "Dict Document"


def test_dvs_db_documents_remove(dvs_client: dvs.DVS):
    # Setup: Create a document to remove
    dvs_client.db.documents.touch()
    doc = Document.from_content(
        content="This document will be removed", name="To Be Removed"
    )
    created_doc = dvs_client.db.documents.create(doc)

    # Verify document exists
    retrieved_doc = dvs_client.db.documents.retrieve(created_doc.document_id)
    assert retrieved_doc.document_id == created_doc.document_id

    # Test: Remove the document
    dvs_client.db.documents.remove(created_doc.document_id)

    # Verify document is gone
    with pytest.raises(NotFoundError):
        dvs_client.db.documents.retrieve(created_doc.document_id)


def test_dvs_db_documents_list(dvs_client: dvs.DVS):
    # Setup: Create test documents
    dvs_client.db.documents.touch()
    docs = []
    for i in range(5):
        doc = Document.from_content(
            content=f"List test document {i}",
            name=f"List Doc {i}",
            metadata={"list_test": True, "index": i},
        )
        docs.append(doc)

    dvs_client.db.documents.bulk_create(docs)

    # Test: List all documents
    result = dvs_client.db.documents.list()
    assert len(result.data) >= 5  # At least our test documents
    assert result.object == "list"
    assert result.first_id is not None
    assert result.last_id is not None

    # Test: List with limit
    limited_result = dvs_client.db.documents.list(limit=3)
    assert len(limited_result.data) == 3

    # Test: List with pagination
    page1 = dvs_client.db.documents.list(limit=2, order="asc")
    assert len(page1.data) == 2

    if page1.has_more:
        page2 = dvs_client.db.documents.list(limit=2, after=page1.last_id, order="asc")
        assert len(page2.data) <= 2
        # Ensure no overlap between pages
        page1_ids = {doc.document_id for doc in page1.data}
        page2_ids = {doc.document_id for doc in page2.data}
        assert page1_ids.isdisjoint(page2_ids)


def test_dvs_db_documents_gen(dvs_client: dvs.DVS):
    # Setup: Create test documents
    dvs_client.db.documents.touch()
    docs = []
    for i in range(4):
        doc = Document.from_content(
            content=f"Generator test document {i}",
            name=f"Gen Doc {i}",
            metadata={"gen_test": True},
        )
        docs.append(doc)

    dvs_client.db.documents.bulk_create(docs)

    # Test: Generate documents
    generated_docs = list(dvs_client.db.documents.gen(limit=2))
    assert len(generated_docs) >= 4  # At least our test documents

    # Test: Generate with limit per page
    gen_with_limit = list(dvs_client.db.documents.gen(limit=1))
    assert len(gen_with_limit) >= 4

    # Verify all generated documents are valid
    for doc in generated_docs:
        assert doc.document_id is not None
        assert doc.name is not None
        assert doc.content is not None


def test_dvs_db_documents_count(dvs_client: dvs.DVS):
    # Setup
    dvs_client.db.documents.touch()

    # Get initial count
    initial_count = dvs_client.db.documents.count()

    # Create test documents
    docs = []
    for i in range(3):
        doc = Document.from_content(
            content=f"Count test document {i}",
            name=f"Count Doc {i}",
            metadata={"count_test": True},
        )
        docs.append(doc)

    created_docs = dvs_client.db.documents.bulk_create(docs)

    # Test: Count all documents
    new_count = dvs_client.db.documents.count()
    assert new_count == initial_count + 3

    # Test: Count by document_id
    count_by_id = dvs_client.db.documents.count(document_id=created_docs[0].document_id)
    assert count_by_id == 1

    # Test: Count by content_md5
    count_by_md5 = dvs_client.db.documents.count(
        content_md5=created_docs[0].content_md5
    )
    assert count_by_md5 == 1

    # Test: Count non-existent document
    count_nonexistent = dvs_client.db.documents.count(document_id="non-existent")
    assert count_nonexistent == 0


def test_dvs_db_documents_content_exists(dvs_client: dvs.DVS):
    # Setup
    dvs_client.db.documents.touch()

    # Create a test document
    doc = Document.from_content(
        content="This content will be checked for existence", name="Content Exists Test"
    )
    created_doc = dvs_client.db.documents.create(doc)

    # Test: Check if content exists
    assert dvs_client.db.documents.content_exists(created_doc.content_md5) is True

    # Test: Check non-existent content
    fake_md5 = "fake_md5_hash_that_does_not_exist"
    assert dvs_client.db.documents.content_exists(fake_md5) is False


def test_dvs_db_documents_drop(dvs_client: dvs.DVS):
    # Setup: Create table and add some documents
    dvs_client.db.documents.touch()
    doc = Document.from_content(
        content="This document will be dropped with the table",
        name="Drop Test Document",
    )
    dvs_client.db.documents.create(doc)

    # Verify document exists
    initial_count = dvs_client.db.documents.count()
    assert initial_count >= 1

    # Test: Drop without force should raise error
    with pytest.raises(ValueError, match="Use force=True to drop table"):
        dvs_client.db.documents.drop()

    # Test: Drop with force
    dvs_client.db.documents.drop(force=True)

    # Verify table is recreated (due to touch_after_drop=True by default)
    # and documents are gone
    new_count = dvs_client.db.documents.count()
    assert new_count == 0

    # Test: Drop without recreating table
    dvs_client.db.documents.create(doc)  # Add a document again
    dvs_client.db.documents.drop(force=True, touch_after_drop=False)

    # The table should not exist, so touch() should recreate it
    dvs_client.db.documents.touch()
    final_count = dvs_client.db.documents.count()
    assert final_count == 0
