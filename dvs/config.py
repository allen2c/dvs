import functools
import pathlib
import typing

import duckdb
import pydantic
import pydantic_settings

if typing.TYPE_CHECKING:
    import rich.console


MANIFEST_TABLE_NAME = "manifest"
POINTS_TABLE_NAME = "points"
DOCUMENTS_TABLE_NAME = "documents"

CACHE_DIR = pathlib.Path("./cache/dvs")
TEMP_DIR = CACHE_DIR.joinpath("tmp")


class Settings(pydantic_settings.BaseSettings):
    """
    Settings for the DuckDB VSS API.
    """

    # DuckDB
    DUCKDB_PATH: typing.Text = pydantic.Field(
        default=...,
        description="The file path to the DuckDB database file containing document and embedding data.",  # noqa: E501
    )

    @property
    def duckdb_conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self.DUCKDB_PATH)

    @functools.cached_property
    def console(self) -> "rich.console.Console":
        import rich.console

        return rich.console.Console()
