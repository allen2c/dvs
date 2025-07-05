import json
import logging
import typing

import jinja2
from openai import NotFoundError

import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.document import Document as DocumentType
from dvs.types.paginations import Pagination
from dvs.utils.display import (
    DISPLAY_SQL_PARAMS,
    DISPLAY_SQL_QUERY,
    display_sql_parameters,
)
from dvs.utils.dummies import dummy_httpx_response
from dvs.utils.sql_stmts import SQL_STMT_DROP_TABLE
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


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
            dur = timer.duration * 1000
            logger.debug(f"Created table: '{dvs.DOCUMENTS_TABLE_NAME}' in {dur:.3f} ms")

        return True

    def retrieve(
        self, document_id: typing.Text, *, verbose: bool | None = None
    ) -> DocumentType:
        """
        Retrieve a document from the DuckDB database by its ID.

        This function queries the database to fetch a document based on the provided
        document ID. It can optionally include the document's embedding vector in the
        result. If the document is not found, a `NotFoundError` is raised.

        Notes
        -----
        - The function uses the model's JSON schema to determine the columns to select.
        - Debug mode provides SQL query details and timing information.
        - The document's metadata is parsed from JSON format before validation.

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> document = Document.objects.retrieve(
        ...     document_id='doc_123',
        ...     conn=conn,
        ...     debug=True
        ... )
        """
        with Timer() as timer:
            out = self._retrieve(document_id, verbose=verbose)
        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Retrieved document: '{document_id}' in {dur:.3f} ms")
        return out

    def create(
        self,
        document: typing.Union[DocumentType, typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> DocumentType:
        """
        Create a single document in the DuckDB database.

        This method wraps the `bulk_create` function to insert a single document into the database. It accepts either a `Document` instance or a dictionary representing the document data. The function returns the created `Document` object.

        Notes
        -----
        - The function uses the `bulk_create` method to handle the insertion, ensuring consistency with batch operations.
        - Debug mode can be enabled to print SQL query details and timing information.

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> document = Document(name='doc_123', content='Sample content')
        >>> created_doc = Point.objects.create(document, conn=conn, debug=True)
        """  # noqa: E501
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            docs = self.bulk_create([document], verbose=verbose)
        doc = docs[0]
        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Created document: '{doc.document_id}' in {dur:.3f} ms")
        return doc

    def bulk_create(
        self,
        documents: typing.Union[
            typing.Sequence[DocumentType],
            typing.Sequence[typing.Dict],
            typing.Sequence[typing.Union[DocumentType, typing.Dict]],
        ],
        *,
        verbose: bool | None = None,
    ) -> typing.List[DocumentType]:
        """
        Insert multiple documents into the DuckDB database.

        This function takes a sequence of documents, which can be either instances of the
        `Document` class or dictionaries, and inserts them into the specified DuckDB table.
        It validates and processes each document according to the model's schema before
        performing the bulk insertion.

        The function supports debugging mode, which provides detailed SQL query information
        and execution timing.

        Notes
        -----
        - The function uses parameterized queries to prevent SQL injection.
        - The execution time is printed in seconds if it exceeds one second, otherwise in milliseconds.
        - The function returns the list of documents that were inserted.
        """  # noqa: E501
        with Timer() as timer:
            documents = [
                (
                    DocumentType.model_validate(doc)
                    if isinstance(doc, typing.Dict)
                    else doc.sanitize()
                )
                for doc in documents
            ]
            self._bulk_create(documents, verbose=verbose)

        if verbose:
            dur, unit = (
                (timer.duration, "s")
                if timer.duration > 1.0
                else (timer.duration * 1000, "ms")
            )
            logger.debug(f"Created {len(documents)} documents in {dur:.3f} {unit}")

        return documents

    def remove(self, document_id: typing.Text, *, verbose: bool | None = None) -> None:
        """
        Remove a document from the DuckDB database by its ID.

        This function executes a DELETE SQL statement to remove a document
        identified by the given `document_id` from the specified DuckDB table.
        It provides an option to output debug information, including the SQL
        query and execution time.

        Notes
        -----
        - The function uses parameterized queries to prevent SQL injection.
        - Debug mode provides SQL query details and timing information.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            self._remove(document_id, verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Deleted document: '{document_id}' in {dur:.3f} ms")
        return None

    def list(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[DocumentType]:
        """
        Retrieve a paginated list of documents from the DuckDB database.

        This function constructs and executes a SQL query to fetch documents from the
        database, with optional filtering based on document ID. It supports pagination
        by allowing the caller to specify a limit on the number of documents returned
        and whether to order the results in ascending or descending order.

        The function also provides an option to output debug information, including
        the SQL query and execution time.

        Notes
        -----
        - The function uses parameterized queries to prevent SQL injection.
        - The `after` and `before` parameters are mutually exclusive and determine
        the starting point for the pagination.
        - The function fetches one more document than the specified limit to check
        if there are more results available.

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> pagination = Document.objects.list(
        ...     after='doc_123',
        ...     limit=10,
        ...     order='asc',
        ...     conn=conn,
        ...     debug=True
        ... )
        """
        with Timer() as timer:
            out = self._list(
                after=after,
                before=before,
                limit=limit,
                order=order,
                verbose=verbose,
            )

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Listed documents in {dur:.3f} ms")
        return out

    def gen(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[DocumentType, None, None]:
        """
        Generate and yield documents from the DuckDB database with pagination support.

        A generator wrapper around the list() method that handles pagination automatically,
        yielding individual documents until all matching records have been retrieved. This is
        useful for processing large result sets without loading all documents into memory at once.

        Notes
        -----
        - Automatically handles pagination using cursor-based pagination with document_id
        - Memory efficient as it yields documents one at a time
        - Maintains the same filtering and ordering capabilities as the list() method

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> for document in Document.objects.gen(
        ...     limit=100,
        ...     conn=conn,
        ...     debug=True
        ... ):
        ...     process_document(document)
        """  # noqa: E501

        has_more = True
        current_after = after
        while has_more:
            documents = self.list(
                after=current_after,
                before=before,
                limit=limit,
                order=order,
                verbose=verbose,
            )
            has_more = documents.has_more
            current_after = documents.last_id
            for doc in documents.data:
                yield doc

    def count(
        self,
        *,
        document_id: typing.Optional[typing.Text] = None,
        content_md5: typing.Optional[typing.Text] = None,
        verbose: bool | None = None,
    ) -> int:
        """
        Count the number of documents in the DuckDB database with optional filters.

        This function executes a SQL COUNT query on the documents table, allowing
        optional filtering by `document_id` and `content_md5`. It provides an option
        to output debug information, including the SQL query and execution time.

        Notes
        -----
        - The function uses parameterized queries to prevent SQL injection.
        - Debug mode provides SQL query details and timing information.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            out = self._count(
                document_id=document_id,
                content_md5=content_md5,
                verbose=verbose,
            )
        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Counted documents in {dur:.3f} ms")
        return out

    def drop(
        self,
        *,
        force: bool = False,
        verbose: bool | None = None,
        touch_after_drop: bool = True,
    ) -> None:
        """
        Drop the documents table from the DuckDB database.

        This method deletes the entire documents table, including all its data and associated
        indexes or constraints. It requires explicit confirmation through the `force` parameter
        to prevent accidental data loss.

        Notes
        -----
        - The operation is irreversible and will permanently delete all data in the table.
        - Debug mode provides SQL query details and timing information.

        Warnings
        --------
        This operation is irreversible and will permanently delete all documents data.
        Use with caution.

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Document.objects.drop(conn=conn, force=True, debug=True)
        Dropping table: 'documents' with SQL:
        ...
        Dropped table: 'documents' in 1.234 ms
        """  # noqa: E501

        if not force:
            raise ValueError("Use force=True to drop table.")

        verbose = self.dvs.verbose if verbose is None else verbose

        with Timer() as timer:
            self._drop(verbose=verbose)

        if touch_after_drop:
            self._touch(verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Dropped table: '{dvs.DOCUMENTS_TABLE_NAME}' in {dur:.3f} ms")

        return None

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

    def _retrieve(
        self, document_id: typing.Text, *, verbose: bool | None = None
    ) -> DocumentType:
        """
        Retrieve a document from the DuckDB database by its ID.
        """
        verbose = self.dvs.verbose if verbose is None else verbose

        columns = list(DocumentType.model_json_schema()["properties"].keys())
        columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = (
            f"SELECT {columns_expr} FROM {dvs.DOCUMENTS_TABLE_NAME} "
            + "WHERE document_id = ?"
        )
        parameters = [document_id]
        if verbose:
            self.dvs.settings.console.print(
                f"\nRetrieving document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = self.dvs.conn.execute(query, parameters).fetchone()

        if result is None:
            raise NotFoundError(
                f"Document with ID '{document_id}' not found.",
                response=dummy_httpx_response(404, b"Not Found"),
                body=None,
            )

        data = dict(zip(columns, result))
        data["metadata"] = json.loads(data["metadata"])
        out = DocumentType.model_validate(data)

        return out

    def _bulk_create(
        self,
        documents: typing.Sequence[DocumentType],
        *,
        verbose: bool | None = None,
    ) -> typing.List[DocumentType]:
        """
        Insert multiple documents into the DuckDB database.
        """  # noqa: E501

        if not documents:
            return []

        columns = list(documents[0].model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: typing.List[typing.Tuple[typing.Any, ...]] = [
            tuple(getattr(doc, c) for c in columns) for doc in documents
        ]

        query = (
            f"INSERT INTO {dvs.DOCUMENTS_TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )
        if verbose:
            _display_params = display_sql_parameters(parameters)
            self.dvs.settings.console.print(
                "\nCreating documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
            )

        # Create documents
        self.dvs.conn.executemany(query, parameters)

        return list(documents)

    def _remove(self, document_id: typing.Text, *, verbose: bool | None) -> None:
        """
        Remove a document from the DuckDB database by its ID.
        """
        # Prepare delete query
        query = f"DELETE FROM {dvs.DOCUMENTS_TABLE_NAME} WHERE document_id = ?"
        parameters = [document_id]
        if verbose:
            self.dvs.settings.console.print(
                f"\nDeleting document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        # Delete document
        self.dvs.conn.execute(query, parameters)

        return None

    def _list(
        self,
        *,
        after: typing.Optional[typing.Text],
        before: typing.Optional[typing.Text],
        limit: int,
        order: typing.Literal["asc", "desc"],
        verbose: bool | None,
    ) -> Pagination[DocumentType]:
        """
        Retrieve a paginated list of documents from the DuckDB database.
        """
        columns = list(DocumentType.model_json_schema()["properties"].keys())
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {dvs.DOCUMENTS_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

        if after is not None and order == "asc":
            where_clauses.append("document_id > ?")
            parameters.append(after)
        elif before is not None and order == "desc":
            where_clauses.append("document_id < ?")
            parameters.append(before)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        query += f"ORDER BY document_id {order.upper()}\n"

        # Fetch one more than the limit to determine if there are more results
        fetch_limit = limit + 1
        query += f"LIMIT {fetch_limit}"

        if verbose:
            self.dvs.settings.console.print(
                "\nListing documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        results = self.dvs.conn.execute(query, parameters).fetchall()
        results = [
            {
                column: (json.loads(value) if column == "metadata" else value)
                for column, value in zip(columns, row)
            }
            for row in results
        ]

        documents = [DocumentType.model_validate(row) for row in results[:limit]]

        out = Pagination.model_validate(
            {
                "data": documents,
                "object": "list",
                "first_id": documents[0].document_id if documents else None,
                "last_id": documents[-1].document_id if documents else None,
                "has_more": len(results) > limit,
            }
        )

        return out

    def _count(
        self,
        *,
        document_id: typing.Optional[typing.Text],
        content_md5: typing.Optional[typing.Text],
        verbose: bool | None,
    ) -> int:
        """
        Count the number of documents in the DuckDB database with optional filters.
        """
        query = f"SELECT COUNT(*) FROM {dvs.DOCUMENTS_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

        if document_id is not None:
            where_clauses.append("document_id = ?")
            parameters.append(document_id)
        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        if verbose:
            self.dvs.settings.console.print(
                "\nCounting documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = self.dvs.conn.execute(query, parameters).fetchone()
        count = result[0] if result else 0

        return count

    def _drop(self, *, verbose: bool | None = None) -> None:
        """
        Drop the documents table from the DuckDB database.
        """  # noqa: E501
        query_template = jinja2.Template(SQL_STMT_DROP_TABLE)
        query = query_template.render(table_name=dvs.DOCUMENTS_TABLE_NAME)

        if verbose:
            self.dvs.settings.console.print(
                f"\nDropping table: '{dvs.DOCUMENTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
            )

        # Drop table
        self.dvs.conn.sql(query)

        return None
