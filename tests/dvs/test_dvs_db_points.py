import dvs


def test_dvs_db_points_touch(dvs_client: dvs.DVS):
    assert dvs_client.db.points.touch()


def test_dvs_db_points_retrieve(dvs_client: dvs.DVS): ...


def test_dvs_db_points_create(dvs_client: dvs.DVS): ...


def test_dvs_db_points_bulk_create(dvs_client: dvs.DVS): ...


def test_dvs_db_points_remove(dvs_client: dvs.DVS): ...


def test_dvs_db_points_list(dvs_client: dvs.DVS): ...


def test_dvs_db_points_gen(dvs_client: dvs.DVS): ...


def test_dvs_db_points_count(dvs_client: dvs.DVS): ...


def test_dvs_db_points_content_exists(dvs_client: dvs.DVS): ...


def test_dvs_db_points_drop(dvs_client: dvs.DVS): ...


def test_dvs_db_points_remove_outdated(dvs_client: dvs.DVS): ...


def test_dvs_db_points_remove_many(dvs_client: dvs.DVS): ...
