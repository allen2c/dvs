import functools
import typing

import dvs
from dvs.utils.display import DISPLAY_SQL_QUERY
from dvs.utils.sql_stmts import (
    SQL_STMT_INSTALL_EXTENSIONS,
)

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

        This method creates the necessary database tables (documents and points) with proper
        schemas and indexes. It installs required DuckDB extensions and sets up HNSW indexing
        for efficient vector similarity searches.

        Notes
        -----
        - Creates 'documents' table for storing document metadata and content
        - Creates 'points' table for storing vector embeddings with HNSW indexing
        - Installs required DuckDB extensions (e.g., JSON, httpfs)
        - Sets up indexes for optimized query performance

        Examples
        --------
        >>> dvs = DVS(duckdb_path="./data/vectors.duckdb")
        >>> dvs.touch(raise_if_exists=True, debug=True)

        Warnings
        --------
        If raise_if_exists=True and tables already exist, raises ConflictError
        with status code 409.
        """  # noqa: E501

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

    @functools.cached_property
    def manifest(self) -> "Manifest":
        from dvs.db.manifest.api import Manifest

        return Manifest(self.dvs)

    @functools.cached_property
    def documents(self) -> "Documents":
        from dvs.db.documents.api import Documents

        return Documents(self.dvs)

    @functools.cached_property
    def points(self) -> "Points":
        from dvs.db.points.api import Points

        return Points(self.dvs)
