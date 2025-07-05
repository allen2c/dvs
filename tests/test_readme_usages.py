"""
Test cases for README code examples.
These tests ensure all code examples in the README work correctly.
"""

import pathlib
import tempfile

import diskcache
import openai
import openai_embeddings_model as oai_emb_model
import pytest

from dvs import DVS
from dvs.types.document import Document


@pytest.mark.asyncio
async def test_readme_basic_usage():
    """Test the basic usage example from README."""
    # Initialize DVS with a database file and model
    dvs = DVS(
        tempfile.NamedTemporaryFile(suffix=".duckdb").name,
        model="text-embedding-3-small",
        model_settings=oai_emb_model.ModelSettings(dimensions=1536),
    )

    # Add documents
    dvs.add("Apple announced new iPhone features with upgraded camera and A16 chip.")
    dvs.add("Microsoft updated Azure with enhanced AI tools and security features.")

    # Search
    results = await dvs.search("What are the new iPhone features?")

    # Verify results
    assert len(results) > 0
    for point, document, score in results:
        assert point is not None
        assert document is not None
        assert isinstance(score, float)


@pytest.mark.asyncio
async def test_readme_advanced_configuration():
    """Test the advanced configuration example from README."""
    # Configure with custom cache and model settings
    dvs = DVS(
        "./test_database.duckdb",
        model=oai_emb_model.OpenAIEmbeddingsModel(
            model="text-embedding-3-small",
            openai_client=openai.OpenAI(),
            cache=diskcache.Cache("./test_cache"),
        ),
        model_settings=oai_emb_model.ModelSettings(dimensions=1536),
        verbose=False,
    )

    # Add documents with metadata
    doc = Document.from_content(
        "Latest developments in artificial intelligence...",
        name="AI Research Paper",
        metadata={"author": "John Doe", "year": 2024},
    )
    dvs.add(doc)

    # Search with more results
    results = await dvs.search("artificial intelligence", top_k=10)

    # Verify results
    assert len(results) > 0

    # Cleanup
    pathlib.Path("./test_database.duckdb").unlink(missing_ok=True)
    import shutil

    shutil.rmtree("./test_cache", ignore_errors=True)


def test_readme_adding_documents():
    """Test the adding documents examples from README."""
    dvs = DVS(
        tempfile.NamedTemporaryFile(suffix=".duckdb").name,
        model="text-embedding-3-small",
        model_settings=oai_emb_model.ModelSettings(dimensions=1536),
    )

    # Add single document
    dvs.add("Your document content here")

    # Add multiple documents
    documents = [
        "First document content",
        "Second document content",
        "Third document content",
    ]
    dvs.add(documents)

    # Add documents with metadata
    docs = [
        Document.from_content("Content 1", name="Doc 1", metadata={"category": "tech"}),
        Document.from_content(
            "Content 2", name="Doc 2", metadata={"category": "science"}
        ),
    ]
    dvs.add(docs)

    # Verify documents were added
    assert True  # If no exception, test passes


@pytest.mark.asyncio
async def test_readme_searching_documents():
    """Test the searching documents examples from README."""
    dvs = DVS(
        tempfile.NamedTemporaryFile(suffix=".duckdb").name,
        model="text-embedding-3-small",
        model_settings=oai_emb_model.ModelSettings(dimensions=1536),
    )

    # Add test documents
    dvs.add("Test document for searching")
    dvs.add("Another test document")

    # Basic search
    results = await dvs.search("test")
    assert len(results) > 0

    # Search with more results
    results = await dvs.search("test", top_k=10)
    assert len(results) > 0

    # Search with embeddings included
    results = await dvs.search("test", with_embedding=True)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_readme_removing_documents():
    """Test the removing documents examples from README."""
    dvs = DVS(
        tempfile.NamedTemporaryFile(suffix=".duckdb").name,
        model="text-embedding-3-small",
        model_settings=oai_emb_model.ModelSettings(dimensions=1536),
    )

    # Add test documents
    dvs.add("Document to be removed")
    dvs.add("Another document to be removed")
    dvs.add("Third document to be removed")

    # Get document ID from search results
    results = await dvs.search("document")
    doc_id = results[0][1].document_id

    # Remove document
    dvs.remove(doc_id)

    # Remove multiple documents
    if len(results) > 1:
        doc_ids = [results[1][1].document_id, results[2][1].document_id]
        dvs.remove(doc_ids)

    # Verify removal worked
    assert True  # If no exception, test passes
