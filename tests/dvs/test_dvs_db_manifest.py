import dvs


def test_dvs_db_manifest_touch(dvs_client: dvs.DVS):
    assert dvs_client.db.manifest.touch()


def test_dvs_db_manifest_receive(dvs_client: dvs.DVS): ...


def test_dvs_db_manifest_create(dvs_client: dvs.DVS): ...
