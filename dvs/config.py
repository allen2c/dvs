import functools
import pathlib
import typing

import duckdb
import pydantic
import pydantic_settings

if typing.TYPE_CHECKING:
    import rich.console


DVS_MANIFEST_TABLE_NAME = "manifest"
DVS_POINTS_TABLE_NAME = "points"
DVS_DOCUMENTS_TABLE_NAME = "documents"
DVS_NODES_TABLE_NAME = "nodes"
DVS_EDGES_IS_A_TABLE_NAME = "edges_is_a"
DVS_EDGES_HAS_A_TABLE_NAME = "edges_has_a"
DVS_EDGES_RELATED_TO_TABLE_NAME = "edges_related_to"
DVS_EDGES_IS_FROM_TABLE_NAME = "edges_is_from"
DVS_GRAPH_TABLE_NAME = "knowledge_graph"

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

    @property
    def duckdb_path(self) -> pathlib.Path:
        return pathlib.Path(self.DUCKDB_PATH)

    @functools.cached_property
    def console(self) -> "rich.console.Console":
        import rich.console

        return rich.console.Console()
