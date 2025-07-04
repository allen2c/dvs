import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.document import Document as DocumentType
from dvs.utils.display import DISPLAY_SQL_QUERY
from dvs.utils.timer import Timer


class Documents:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        """
        Ensure the existence of the documents table in the DuckDB database.

        This method checks if the documents table exists in the database. If it does not exist,
        it creates the table using the model's JSON schema. If the table already exists and
        `raise_if_exists` is set to True, a `ConflictError` is raised. The method also installs
        necessary JSON and VSS extensions before creating the table.

        Notes
        -----
        - The table is created with `document_id` as the primary key and an index on `content_md5`.
        - Debug mode provides SQL query details and timing information.
        - The function returns True upon successful execution.

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Document.objects.touch(conn=conn, raise_if_exists=True, debug=True)
        """  # noqa: E501
        with Timer() as timer:
            self._touch(verbose=verbose)

        if verbose:
            self.dvs.settings.console.print(
                f"Created table: '{dvs.DOCUMENTS_TABLE_NAME}' in "
                + f"{timer.duration:.3f} ms"
            )

        return True

    def _touch(self, *, verbose: bool | None = None) -> bool:
        """
        Ensure the existence of the documents table in the DuckDB database.
        """
        # Install JSON and VSS extensions
        self.dvs.db.install_extensions(verbose=verbose)

        # Create table
        create_table_sql = openapi_utils.openapi_to_create_table_sql(
            DocumentType.model_json_schema(),
            table_name=dvs.DOCUMENTS_TABLE_NAME,
            primary_key="document_id",
            unique_fields=[],
            # unique_fields=["name"],  # Index limitations (https://duckdb.org/docs/sql/indexes)  # noqa: E501
            indexes=["content_md5"],
        )
        if verbose:
            self.dvs.settings.console.print(
                f"\nCreating table: '{dvs.DOCUMENTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )

        self.dvs.conn.sql(create_table_sql)

        return True
