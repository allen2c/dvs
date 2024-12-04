from pathlib import Path
from typing import Optional

import duckdb

from dvs.config import settings
from dvs.types.document import Document
from dvs.types.point import Point


class DVS:
    def __init__(
        self,
        duckdb_path: Optional[Path] = None,
        *,
        touch: bool = True,
        raise_if_exists: bool = False,
        debug: bool = False
    ):
        self._db_path = duckdb_path or Path(settings.DUCKDB_PATH)
        self.debug = debug

        if touch:
            self.touch(raise_if_exists=raise_if_exists, debug=debug)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path)  # Always open a new duckdb connection

    def touch(self, *, raise_if_exists: bool = False, debug: bool = False):
        Document.objects.touch(
            conn=self.conn, raise_if_exists=raise_if_exists, debug=debug
        )
        Point.objects.touch(
            conn=self.conn, raise_if_exists=raise_if_exists, debug=debug
        )
