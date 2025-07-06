import functools
import typing

import dvs
from dvs.utils.display import DISPLAY_SQL_QUERY
from dvs.utils.sql_stmts import SQL_STMT_INSTALL_EXTENSIONS, SQL_STMT_SHOW_TABLES

if typing.TYPE_CHECKING:
    from dvs.db.documents.api import Documents
    from dvs.db.manifest.api import Manifest
    from dvs.db.points.api import Points


class DB:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        """
        Initialize the DuckDB database tables required for vector similarity search.
        Creates manifest, documents, and points tables with proper schemas and indexes.
        Installs required DuckDB extensions and sets up HNSW indexing for searches.
        """
        if not self.manifest.touch(verbose=verbose):
            raise ValueError("Failed to touch the manifest table")
        if not self.documents.touch(verbose=verbose):
            raise ValueError("Failed to touch the documents table")
        if not self.points.touch(verbose=verbose):
            raise ValueError("Failed to touch the points table")
        return True

    def install_extensions(self, *, verbose: bool | None = None) -> bool:
        """
        Install required DuckDB extensions for the database.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        if verbose:
            self.dvs.settings.console.print(
                "\nInstalling extensions with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=SQL_STMT_INSTALL_EXTENSIONS)}\n"
            )
        self.dvs.conn.sql(SQL_STMT_INSTALL_EXTENSIONS)

        return True

    def show_table_names(self) -> typing.Tuple[typing.Text, ...]:
        """
        Return the names of all tables in the database.
        """
        res: typing.List[typing.Tuple[typing.Text]] = self.dvs.conn.execute(
            SQL_STMT_SHOW_TABLES
        ).fetchall()
        return tuple(r[0] for r in res)

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
