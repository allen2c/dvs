import dvs


def test_dvs_db_documents_touch(dvs_client: dvs.DVS):
    assert dvs_client.db.documents.touch()


def test_dvs_db_documents_retrieve(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_create(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_bulk_create(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_remove(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_list(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_gen(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_count(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_content_exists(dvs_client: dvs.DVS): ...


def test_dvs_db_documents_drop(dvs_client: dvs.DVS): ...
