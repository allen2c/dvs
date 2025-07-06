import pytest
from openai import NotFoundError

import dvs
from dvs.types.point import Point


def _create_test_embedding(seed: int = 0) -> list[float]:
    """Create a test embedding with 512 dimensions."""
    import random

    random.seed(seed)
    return [random.uniform(-1.0, 1.0) for _ in range(512)]


def test_dvs_db_points_touch(dvs_client: dvs.DVS):
    assert dvs_client.db.points.touch()


def test_dvs_db_points_retrieve(dvs_client: dvs.DVS):
    # Setup: Create a point to retrieve
    dvs_client.db.points.touch()
    point = Point(
        document_id="doc_123",
        content_md5="test_hash_123",
        embedding=_create_test_embedding(1),  # type: ignore
        metadata={"test": "retrieve"},
    )
    created_point = dvs_client.db.points.create(point)

    # Test: Retrieve the point
    retrieved_point = dvs_client.db.points.retrieve(created_point.point_id)

    # Assertions
    assert retrieved_point.point_id == created_point.point_id
    assert retrieved_point.document_id == "doc_123"
    assert retrieved_point.content_md5 == "test_hash_123"
    assert retrieved_point.metadata == {"test": "retrieve"}

    # Test: Retrieve with embedding
    retrieved_with_embedding = dvs_client.db.points.retrieve(
        created_point.point_id, with_embedding=True
    )
    assert retrieved_with_embedding.embedding is not None

    # Test: Retrieve non-existent point should raise NotFoundError
    with pytest.raises(NotFoundError):
        dvs_client.db.points.retrieve("non-existent-id")


def test_dvs_db_points_create(dvs_client: dvs.DVS):
    # Setup
    dvs_client.db.points.touch()

    # Test: Create from Point object
    point = Point(
        document_id="doc_456",
        content_md5="test_hash_456",
        embedding=_create_test_embedding(2),  # type: ignore
        metadata={"test": "create", "index": 1},
    )
    created_point = dvs_client.db.points.create(point)

    # Assertions
    assert created_point.point_id is not None
    assert created_point.document_id == "doc_456"
    assert created_point.content_md5 == "test_hash_456"
    assert created_point.metadata == {"test": "create", "index": 1}
    assert created_point.embedding is not None

    # Test: Create from dictionary
    point_dict = {
        "document_id": "doc_789",
        "content_md5": "test_hash_789",
        "embedding": _create_test_embedding(3),
        "metadata": {"test": "dict"},
    }
    created_from_dict = dvs_client.db.points.create(point_dict)

    # Assertions
    assert created_from_dict.point_id is not None
    assert created_from_dict.document_id == "doc_789"
    assert created_from_dict.content_md5 == "test_hash_789"
    assert created_from_dict.metadata == {"test": "dict"}


def test_dvs_db_points_bulk_create(dvs_client: dvs.DVS):
    # Setup
    dvs_client.db.points.touch()

    # Test: Create multiple points
    points = []
    for i in range(3):
        point = Point(
            document_id=f"doc_bulk_{i}",
            content_md5=f"hash_bulk_{i}",
            embedding=_create_test_embedding(10 + i),  # type: ignore
            metadata={"test": "bulk", "index": i},
        )
        points.append(point)

    created_points = dvs_client.db.points.bulk_create(points)

    # Assertions
    assert len(created_points) == 3
    for i, point in enumerate(created_points):
        assert point.point_id is not None
        assert point.document_id == f"doc_bulk_{i}"
        assert point.content_md5 == f"hash_bulk_{i}"
        assert point.metadata == {"test": "bulk", "index": i}

    # Test: Create from mixed types (Points and dicts)
    mixed_points = [
        Point(
            document_id="doc_mixed_1",
            content_md5="hash_mixed_1",
            embedding=_create_test_embedding(20),  # type: ignore
            metadata={"type": "point"},
        ),
        {
            "document_id": "doc_mixed_2",
            "content_md5": "hash_mixed_2",
            "embedding": _create_test_embedding(21),
            "metadata": {"type": "dict"},
        },
    ]

    created_mixed = dvs_client.db.points.bulk_create(mixed_points)
    assert len(created_mixed) == 2
    assert created_mixed[0].document_id == "doc_mixed_1"
    assert created_mixed[1].document_id == "doc_mixed_2"


def test_dvs_db_points_remove(dvs_client: dvs.DVS):
    # Setup: Create a point to remove
    dvs_client.db.points.touch()
    point = Point(
        document_id="doc_remove",
        content_md5="hash_remove",
        embedding=_create_test_embedding(30),  # type: ignore
        metadata={"test": "remove"},
    )
    created_point = dvs_client.db.points.create(point)

    # Verify point exists
    retrieved_point = dvs_client.db.points.retrieve(created_point.point_id)
    assert retrieved_point.point_id == created_point.point_id

    # Test: Remove the point
    dvs_client.db.points.remove(created_point.point_id)

    # Verify point is gone
    with pytest.raises(NotFoundError):
        dvs_client.db.points.retrieve(created_point.point_id)


def test_dvs_db_points_list(dvs_client: dvs.DVS):
    # Setup: Create test points
    dvs_client.db.points.touch()
    points = []
    for i in range(5):
        point = Point(
            document_id=f"doc_list_{i}",
            content_md5=f"hash_list_{i}",
            embedding=_create_test_embedding(40 + i),  # type: ignore
            metadata={"test": "list", "index": i},
        )
        points.append(point)

    dvs_client.db.points.bulk_create(points)

    # Test: List all points
    result = dvs_client.db.points.list()
    assert len(result.data) >= 5  # At least our test points
    assert result.object == "list"
    assert result.first_id is not None
    assert result.last_id is not None

    # Test: List with limit
    limited_result = dvs_client.db.points.list(limit=3)
    assert len(limited_result.data) == 3

    # Test: List with document_id filter
    filtered_result = dvs_client.db.points.list(document_id="doc_list_0")
    assert len(filtered_result.data) == 1
    assert filtered_result.data[0].document_id == "doc_list_0"

    # Test: List with content_md5 filter
    filtered_result = dvs_client.db.points.list(content_md5="hash_list_1")
    assert len(filtered_result.data) == 1
    assert filtered_result.data[0].content_md5 == "hash_list_1"

    # Test: List with pagination
    page1 = dvs_client.db.points.list(limit=2, order="asc")
    assert len(page1.data) == 2

    if page1.has_more:
        page2 = dvs_client.db.points.list(limit=2, after=page1.last_id, order="asc")
        assert len(page2.data) <= 2
        # Ensure no overlap between pages
        page1_ids = {point.point_id for point in page1.data}
        page2_ids = {point.point_id for point in page2.data}
        assert page1_ids.isdisjoint(page2_ids)


def test_dvs_db_points_gen(dvs_client: dvs.DVS):
    # Setup: Create test points
    dvs_client.db.points.touch()
    points = []
    for i in range(4):
        point = Point(
            document_id=f"doc_gen_{i}",
            content_md5=f"hash_gen_{i}",
            embedding=_create_test_embedding(50 + i),  # type: ignore
            metadata={"test": "gen", "index": i},
        )
        points.append(point)

    dvs_client.db.points.bulk_create(points)

    # Test: Generate points
    generated_points = list(dvs_client.db.points.gen(limit=2))
    assert len(generated_points) >= 4  # At least our test points

    # Test: Generate with limit per page
    gen_with_limit = list(dvs_client.db.points.gen(limit=1))
    assert len(gen_with_limit) >= 4

    # Test: Generate with filters
    gen_filtered = list(dvs_client.db.points.gen(document_id="doc_gen_0"))
    assert len(gen_filtered) == 1
    assert gen_filtered[0].document_id == "doc_gen_0"

    # Verify all generated points are valid
    for point in generated_points:
        assert point.point_id is not None
        assert point.document_id is not None
        assert point.content_md5 is not None


def test_dvs_db_points_count(dvs_client: dvs.DVS):
    # Setup: Create test points
    dvs_client.db.points.touch()
    points = []
    for i in range(3):
        point = Point(
            document_id=f"doc_count_{i}",
            content_md5=f"hash_count_{i}",
            embedding=_create_test_embedding(60 + i),  # type: ignore
            metadata={"test": "count"},
        )
        points.append(point)

    dvs_client.db.points.bulk_create(points)

    # Test: Count all points
    total_count = dvs_client.db.points.count()
    assert total_count >= 3  # At least our test points

    # Test: Count with document_id filter
    doc_count = dvs_client.db.points.count(document_id="doc_count_0")
    assert doc_count == 1

    # Test: Count with content_md5 filter
    content_count = dvs_client.db.points.count(content_md5="hash_count_1")
    assert content_count == 1

    # Test: Count with non-existent filter
    no_count = dvs_client.db.points.count(document_id="non_existent")
    assert no_count == 0


def test_dvs_db_points_content_exists(dvs_client: dvs.DVS):
    # Setup: Create a point
    dvs_client.db.points.touch()
    point = Point(
        document_id="doc_exists",
        content_md5="hash_exists",
        embedding=_create_test_embedding(70),  # type: ignore
        metadata={"test": "exists"},
    )
    created_point = dvs_client.db.points.create(point)

    # Test: Check if content exists
    assert dvs_client.db.points.content_exists(created_point.content_md5) is True

    # Test: Check if non-existent content exists
    fake_md5 = "fake_hash_does_not_exist"
    assert dvs_client.db.points.content_exists(fake_md5) is False


def test_dvs_db_points_drop(dvs_client: dvs.DVS):
    # Setup: Create table and add some points
    dvs_client.db.points.touch()
    point = Point(
        document_id="doc_drop",
        content_md5="hash_drop",
        embedding=_create_test_embedding(80),  # type: ignore
        metadata={"test": "drop"},
    )
    dvs_client.db.points.create(point)

    # Verify point exists
    count_before = dvs_client.db.points.count()
    assert count_before >= 1

    # Test: Drop table (requires force=True)
    dvs_client.db.points.drop(force=True)

    # Test: Verify table is recreated and empty
    count_after = dvs_client.db.points.count()
    assert count_after == 0


def test_dvs_db_points_remove_outdated(dvs_client: dvs.DVS):
    # Setup: Create points with same document_id but different content_md5
    dvs_client.db.points.touch()

    # Create old point
    old_point = Point(
        document_id="doc_outdated",
        content_md5="old_hash",
        embedding=_create_test_embedding(90),  # type: ignore
        metadata={"version": "old"},
    )
    dvs_client.db.points.create(old_point)

    # Create new point with same document_id but different content_md5
    new_point = Point(
        document_id="doc_outdated",
        content_md5="new_hash",
        embedding=_create_test_embedding(91),  # type: ignore
        metadata={"version": "new"},
    )
    dvs_client.db.points.create(new_point)

    # Verify both points exist
    assert dvs_client.db.points.count(document_id="doc_outdated") == 2

    # Test: Remove outdated points (keep only the new content)
    dvs_client.db.points.remove_outdated(
        document_id="doc_outdated", content_md5="new_hash"
    )

    # Verify only the new point remains
    remaining_points = dvs_client.db.points.list(document_id="doc_outdated")
    assert len(remaining_points.data) == 1
    assert remaining_points.data[0].content_md5 == "new_hash"


def test_dvs_db_points_remove_many(dvs_client: dvs.DVS):
    # Setup: Create multiple points
    dvs_client.db.points.touch()
    points = []
    for i in range(5):
        point = Point(
            document_id=f"doc_many_{i}",
            content_md5=f"hash_many_{i}",
            embedding=_create_test_embedding(100 + i),  # type: ignore
            metadata={"test": "remove_many", "index": i},
        )
        points.append(point)

    created_points = dvs_client.db.points.bulk_create(points)

    # Test: Remove by point_ids
    point_ids_to_remove = [created_points[0].point_id, created_points[1].point_id]
    dvs_client.db.points.remove_many(point_ids=point_ids_to_remove)

    # Verify points are removed
    for point_id in point_ids_to_remove:
        with pytest.raises(NotFoundError):
            dvs_client.db.points.retrieve(point_id)

    # Test: Remove by document_ids
    document_ids_to_remove = ["doc_many_2", "doc_many_3"]
    dvs_client.db.points.remove_many(document_ids=document_ids_to_remove)

    # Verify points are removed
    for doc_id in document_ids_to_remove:
        remaining_points = dvs_client.db.points.list(document_id=doc_id)
        assert len(remaining_points.data) == 0

    # Test: Remove by content_md5s
    content_md5s_to_remove = ["hash_many_4"]
    dvs_client.db.points.remove_many(content_md5s=content_md5s_to_remove)

    # Verify points are removed
    for content_md5 in content_md5s_to_remove:
        remaining_points = dvs_client.db.points.list(content_md5=content_md5)
        assert len(remaining_points.data) == 0
