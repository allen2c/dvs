import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.point import Point as PointType
from dvs.utils.display import DISPLAY_SQL_QUERY
from dvs.utils.sql_stmts import (
    SQL_STMT_CREATE_EMBEDDING_INDEX,
    SQL_STMT_INSTALL_EXTENSIONS,
    SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE,
)
from dvs.utils.timer import Timer


class Points:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        """
        Initialize the points table in DuckDB with required extensions and indexes for vector similarity search.

        This method creates the points table with proper schema derived from the Point model,
        sets up HNSW indexing for vector similarity search, and installs necessary DuckDB
        extensions. The table structure includes columns for point_id (primary key),
        document_id, content_md5, and embedding vectors.

        Notes
        -----
        - Creates indexes on document_id and content_md5 columns for faster lookups
        - Installs JSON extension for metadata handling
        - Sets up HNSW (Hierarchical Navigable Small World) index for efficient vector search
        - Enables experimental HNSW persistence for index durability

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Point.objects.touch(conn=conn, debug=True)
        Creating table: 'points' with SQL:
        ...
        Created table: 'points' in 123.456 ms
        True

        Warnings
        --------
        If raise_if_exists=True and the table already exists, raises ConflictError
        with status code 409.
        """  # noqa: E501
        with Timer() as timer:
            self._touch(verbose=verbose)
        if verbose:
            self.dvs.settings.console.print(
                f"Created table: '{dvs.POINTS_TABLE_NAME}' in "
                + f"{timer.duration:.3f} ms"
            )
        return True

    def _touch(self, *, verbose: bool | None = None) -> bool:
        """
        Initialize the points table in DuckDB with required extensions and indexes for vector similarity search.
        """  # noqa: E501
        # Install JSON and VSS extensions
        self.dvs.db.install_extensions(verbose=verbose)

        # Create table
        create_table_sql = openapi_utils.openapi_to_create_table_sql(
            PointType.model_json_schema(),
            table_name=dvs.POINTS_TABLE_NAME,
            primary_key="point_id",
            indexes=["document_id", "content_md5"],
        ).strip()
        create_table_sql = (
            SQL_STMT_INSTALL_EXTENSIONS
            + f"\n{create_table_sql}\n"
            #
            + f"\n{SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE}\n"
            # Required for HNSW index
            #
            + SQL_STMT_CREATE_EMBEDDING_INDEX.format(
                table_name=dvs.POINTS_TABLE_NAME,
                column_name="embedding",
                metric="cosine",
            )
        ).strip()

        if verbose:
            self.dvs.settings.console.print(
                f"\nCreating table: '{dvs.POINTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )
        self.dvs.conn.sql(create_table_sql)

        return True
