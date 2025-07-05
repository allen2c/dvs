import pytest

import dvs


def test_dvs_db_touch(dvs_client: dvs.DVS):
    assert dvs_client.db.touch()


def test_dvs_add(dvs_client: dvs.DVS): ...


def test_dvs_remove(dvs_client: dvs.DVS): ...


@pytest.mark.asyncio
async def test_dvs_search(dvs_client: dvs.DVS): ...
