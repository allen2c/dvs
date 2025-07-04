import json
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    List,
    Literal,
    Optional,
    Sequence,
    Text,
    Tuple,
    Type,
    Union,
)

import duckdb
import jinja2
from openai import APIStatusError, ConflictError, NotFoundError
from tqdm import tqdm

import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.document import Document as DocumentType
from dvs.types.paginations import Pagination
from dvs.utils.chunk import chunks
from dvs.utils.display import (
    DISPLAY_SQL_PARAMS,
    DISPLAY_SQL_QUERY,
    display_sql_parameters,
)
from dvs.utils.dummies import dummy_httpx_response
from dvs.utils.ensure import ensure_dict
from dvs.utils.openapi import openapi_to_create_table_sql
from dvs.utils.sql_stmts import (
    SQL_STMT_CREATE_EMBEDDING_INDEX,
    SQL_STMT_DROP_TABLE,
    SQL_STMT_INSTALL_EXTENSIONS,
    SQL_STMT_REMOVE_OUTDATED_POINTS,
    SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE,
    SQL_STMT_SHOW_TABLES,
)
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

    def retrieve(
        self, document_id: Text, *, verbose: bool | None = None
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
            self.dvs.settings.console.print(
                f"Retrieved document: '{document_id}' in " + f"{timer.duration:.3f} ms"
            )
        return out

    def create(
        self, document: Union[DocumentType, Dict], *, verbose: bool | None = None
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
            docs = self._bulk_create([document], verbose=verbose)  # type: ignore
        doc = docs[0]
        if verbose:
            self.dvs.settings.console.print(
                f"Created document: '{doc.document_id}' in "
                + f"{timer.duration:.3f} ms"
            )
        return doc

    def bulk_create(
        self,
        documents: Union[
            Sequence[DocumentType], Sequence[Dict], Sequence[Union[DocumentType, Dict]]
        ],
        *,
        verbose: bool | None = None,
    ) -> List[DocumentType]:
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
                    DocumentType.model_validate(doc).strip()
                    if isinstance(doc, Dict)
                    else doc.strip()
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
            self.dvs.settings.console.print(
                f"Created {len(documents)} documents in {dur:.3f} {unit}"
            )

        return documents

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
        self, document_id: Text, *, verbose: bool | None = None
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
        self, documents: Sequence["DocumentType"], *, verbose: bool | None = None
    ) -> List[DocumentType]:
        """
        Insert multiple documents into the DuckDB database.
        """  # noqa: E501

        if not documents:
            return []

        columns = list(documents[0].model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: List[Tuple[Any, ...]] = [
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
