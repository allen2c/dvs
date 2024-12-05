import pathlib
import typing

import pytest

from dvs import DVS
from dvs.config import settings
from dvs.types.document import Document


@pytest.mark.asyncio
async def test_dvs(documents: typing.List[Document]):
    pathlib.Path(settings.DUCKDB_PATH).unlink(missing_ok=True)  # clean up

    # Initialize DVS
    client = DVS()

    # Add documents
    documents_with_points = client.add(documents, debug=True)
    for doc, pts in documents_with_points:
        assert doc.document_id is not None
        assert len(pts) > 0
        assert all(pt.document_id == doc.document_id for pt in pts)

    # Search
    search_results = await client.search("Sony Playstation 5", debug=True)
    assert len(search_results) > 0
    top_pt, top_doc, top_score = search_results[0]
    assert top_pt is not None
    assert top_doc is not None
    assert top_score is not None
    assert top_pt.document_id == top_doc.document_id
    top_doc_id = top_doc.document_id
    for result in search_results:
        _pt, _doc, _score = result
        assert _pt is not None
        assert _doc is not None
        assert _score is not None
        assert _pt.document_id == _doc.document_id

    # Remove documents
    client.remove(top_doc.document_id, debug=True)
    search_results = await client.search("Sony Playstation 5", debug=True)
    assert len(search_results) > 0
    top_pt, top_doc, top_score = search_results[0]
    assert top_pt is not None
    assert top_doc is not None
    assert top_score is not None
    assert top_pt.document_id != top_doc_id  # Original top document was removed
