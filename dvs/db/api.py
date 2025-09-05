import functools
import typing

import duckdb

import dvs
from dvs.utils.debug_print import debug_print
from dvs.utils.sql_stmts import SQL_STMT_INSTALL_EXTENSIONS, SQL_STMT_SHOW_TABLES
from dvs.utils.timer import Timer

if typing.TYPE_CHECKING:
    from dvs.db.documents.api import Documents
    from dvs.db.graph.api import Graph
    from dvs.db.manifest.api import Manifest
    from dvs.db.points.api import Points


class DB:
    def __init__(self, dvs: dvs.DVS):
        """Initialize database API with DVS instance."""
        self.dvs = dvs

    def touch(self, *, enable_graph: bool = False, verbose: bool | None = None) -> bool:
        """
        Initialize the DuckDB database tables required for vector similarity search.
        Creates manifest, documents, and points tables with proper schemas and indexes.
        Installs required DuckDB extensions and sets up HNSW indexing for searches.
        """
        if not self.manifest.touch(verbose=self.dvs.v(verbose)):
            raise ValueError("Failed to touch the manifest table")
        if not self.documents.touch(verbose=self.dvs.v(verbose)):
            raise ValueError("Failed to touch the documents table")
        if not self.points.touch(verbose=self.dvs.v(verbose)):
            raise ValueError("Failed to touch the points table")
        if enable_graph and not self.graph.touch(verbose=self.dvs.v(verbose)):
            raise ValueError("Failed to touch the graph table")
        return True

    def install_extensions(self, *, verbose: bool | None = None) -> bool:
        """
        Install required DuckDB extensions for the database.
        """

        with Timer() as timer:
            self.dvs.new_connection().cursor().sql(SQL_STMT_INSTALL_EXTENSIONS)

        debug_print(
            SQL_STMT_INSTALL_EXTENSIONS,
            title="Installing extensions with SQL",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=self.dvs.v(verbose),
        )

        return True

    def show_table_names(self) -> typing.Tuple[typing.Text, ...]:
        """
        Return the names of all tables in the database.
        """
        try:
            res: typing.List[typing.Tuple[typing.Text]] = (
                self.dvs.new_connection(read_only=True)
                .cursor()
                .execute(SQL_STMT_SHOW_TABLES)
                .fetchall()
            )
            return tuple(r[0] for r in res)
        except duckdb.IOException as e:
            if "database does not exist" in str(e).lower():
                return tuple()
            else:
                raise e

    @functools.cached_property
    def manifest(self) -> "Manifest":
        """
        Access the manifest table API for managing database metadata.
        """
        from dvs.db.manifest.api import Manifest

        return Manifest(self.dvs)

    @functools.cached_property
    def documents(self) -> "Documents":
        """
        Access the documents table API for managing document storage.
        """
        from dvs.db.documents.api import Documents

        return Documents(self.dvs)

    @functools.cached_property
    def points(self) -> "Points":
        """
        Access the points table API for managing vector embeddings.
        """
        from dvs.db.points.api import Points

        return Points(self.dvs)

    @functools.cached_property
    def graph(self) -> "Graph":
        """
        Access the graph table API for managing graph data.
        """
        from dvs.db.graph.api import Graph

        return Graph(self.dvs)
