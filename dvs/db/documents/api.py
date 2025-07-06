# dvs/db/documents/api.py
import json
import logging
import typing

import duckdb
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
        Creates the table if it doesn't exist, installs necessary extensions.
        """
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
        Raises NotFoundError if the document doesn't exist.
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
        Accepts either a Document instance or a dictionary.
        """
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
        Validates and processes each document before bulk insertion.
        """
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
        Uses parameterized queries to prevent SQL injection.
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
        content_md5: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[DocumentType]:
        """
        Retrieve a paginated list of documents from the DuckDB database.
        Supports filtering by content_md5 and cursor-based pagination.
        """
        with Timer() as timer:
            out = self._list(
                content_md5=content_md5,
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
        content_md5: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[DocumentType, None, None]:
        """
        Generate and yield documents from the DuckDB database with pagination support.
        Memory efficient generator that automatically handles pagination.
        """
        has_more = True
        current_after = after
        while has_more:
            documents = self._list(
                content_md5=content_md5,
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
        Count the number of documents in the DuckDB database.
        Supports optional filtering by document_id and content_md5.
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

    def content_exists(
        self, content_md5: typing.Text, *, verbose: bool | None = None
    ) -> bool:
        """
        Check if a document with the given content_md5 exists in the database.
        """
        with Timer() as timer:
            out = self._list(
                content_md5=content_md5,
                limit=1,
                order="asc",
                verbose=verbose,
                after=None,
                before=None,
            )
        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Checked content_md5 in {dur:.3f} ms")
        return len(out.data) > 0

    def drop(
        self,
        *,
        force: bool = False,
        verbose: bool | None = None,
        touch_after_drop: bool = True,
    ) -> None:
        """
        Drop the documents table from the DuckDB database.
        Requires force=True to prevent accidental data loss.
        Optionally recreates the table after dropping.
        """
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
            indexes=["content_md5", "source_id"],
        )
        if verbose:
            self.dvs.settings.console.print(
                f"\nCreating table: '{dvs.DOCUMENTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )

        try:
            self.dvs.conn.sql(create_table_sql)
        except duckdb.CatalogException as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Table '{dvs.DOCUMENTS_TABLE_NAME}' already exists")
            else:
                raise e

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
        content_md5: typing.Optional[typing.Text],
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

        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

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
