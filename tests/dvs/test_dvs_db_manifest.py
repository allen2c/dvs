import dvs
from dvs.types.manifest import Manifest


def test_dvs_db_manifest_touch(dvs_client: dvs.DVS):
    # Setup: Drop manifest table to start with clean state
    dvs_client.db.manifest.drop()

    assert dvs_client.db.manifest.touch()


def test_dvs_db_manifest_receive(dvs_client: dvs.DVS):
    # Setup: Drop manifest table to start with clean state
    dvs_client.db.manifest.drop()

    # Setup: Create manifest table
    dvs_client.db.manifest.touch()

    # Test: Receive when no manifest exists
    result = dvs_client.db.manifest.receive()
    assert result is None

    # Setup: Create a manifest
    manifest = Manifest(
        name="test_database",
        description="Test database for manifest tests",
        embedding_model="text-embedding-3-small",
        embedding_dimensions=1536,
    )
    dvs_client.db.manifest.create(manifest)

    # Test: Receive existing manifest
    retrieved_manifest = dvs_client.db.manifest.receive()
    assert retrieved_manifest is not None
    assert retrieved_manifest.name == "test_database"
    assert retrieved_manifest.description == "Test database for manifest tests"
    assert retrieved_manifest.embedding_model == "text-embedding-3-small"
    assert retrieved_manifest.embedding_dimensions == 1536


def test_dvs_db_manifest_create(dvs_client: dvs.DVS):
    # Setup: Drop manifest table to start with clean state
    dvs_client.db.manifest.drop()

    # Setup: Create manifest table
    dvs_client.db.manifest.touch()

    # Test: Create manifest
    manifest = Manifest(
        name="sample_database",
        description="Sample database for testing",
        embedding_model="text-embedding-3-large",
        embedding_dimensions=3072,
    )

    created_manifest = dvs_client.db.manifest.create(manifest)

    # Assertions
    assert created_manifest.name == "sample_database"
    assert created_manifest.description == "Sample database for testing"
    assert created_manifest.embedding_model == "text-embedding-3-large"
    assert created_manifest.embedding_dimensions == 3072
    assert created_manifest.version == dvs.__version__
    assert created_manifest.manifest_table_name == "manifest"
    assert created_manifest.points_table_name == "points"
    assert created_manifest.documents_table_name == "documents"

    # Test: Verify manifest was stored in database
    retrieved_manifest = dvs_client.db.manifest.receive()
    assert retrieved_manifest is not None
    assert retrieved_manifest.name == "sample_database"
    assert retrieved_manifest.description == "Sample database for testing"
    assert retrieved_manifest.embedding_model == "text-embedding-3-large"
    assert retrieved_manifest.embedding_dimensions == 3072
