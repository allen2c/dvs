import json
import time
from textwrap import dedent
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
import httpx
import jinja2
from openai import APIStatusError, ConflictError, NotFoundError

from dvs.config import console, settings
from dvs.types.paginations import Pagination
from dvs.utils.display import (
    DISPLAY_SQL_PARAMS,
    DISPLAY_SQL_QUERY,
    display_sql_parameters,
)
from dvs.utils.openapi import openapi_to_create_table_sql
from dvs.utils.sql_stmts import (
    SQL_STMT_CREATE_EMBEDDING_INDEX,
    SQL_STMT_DROP_TABLE,
    SQL_STMT_REMOVE_OUTDATED_POINTS,
    SQL_STMT_SHOW_TABLES,
)

if TYPE_CHECKING:
    from dvs.types.document import Document
    from dvs.types.point import Point


def show_tables(conn: "duckdb.DuckDBPyConnection") -> Tuple[Text, ...]:
    res: List[Tuple[Text]] = conn.sql(SQL_STMT_SHOW_TABLES).fetchall()
    return tuple(r[0] for r in res)


def install_extensions(
    conn: "duckdb.DuckDBPyConnection", *, debug: bool = False
) -> None:
    sql_stmt = dedent(
        """
        INSTALL vss;
        LOAD vss;
        INSTALL json;
        LOAD json;
        """
    ).strip()
    if debug:
        console.print(
            "\nInstalling extensions with SQL:\n"
            + f"{DISPLAY_SQL_QUERY.format(sql=sql_stmt)}\n"
        )
    conn.sql(sql_stmt)


class PointQuerySet:
    def __init__(
        self,
        model: Type["Point"],
        *args,
        **kwargs,
    ):
        self.model = model
        self.__args = args
        self.__kwargs = kwargs

    def touch(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        raise_if_exists: bool = False,
        debug: bool = False,
    ) -> bool:
        """"""

        time_start = time.perf_counter() if debug else None

        # Check if table exists
        if (
            settings.POINTS_TABLE_NAME in show_tables(conn=conn)
            and raise_if_exists is True
        ):
            raise ConflictError(
                f"Table '{settings.POINTS_TABLE_NAME}' already exists.",
                response=httpx.Response(status_code=409),
                body=None,
            )

        # Install JSON and VSS extensions
        install_extensions(conn=conn, debug=debug)

        # Create table
        create_table_sql = openapi_to_create_table_sql(
            self.model.model_json_schema(),
            table_name=settings.POINTS_TABLE_NAME,
            primary_key="point_id",
            indexes=["document_id", "content_md5"],
        ).strip()
        create_table_sql = (
            "INSTALL vss;\n"
            + "LOAD vss;\n"
            + "INSTALL json;\n"
            + "LOAD json;\n"
            + f"\n{create_table_sql}\n"
            + "\nSET hnsw_enable_experimental_persistence = true;\n"  # Required for HNSW index  # noqa: E501
            + SQL_STMT_CREATE_EMBEDDING_INDEX.format(
                table_name=settings.POINTS_TABLE_NAME,
                column_name="embedding",
                metric="cosine",
            )
        ).strip()

        if debug:
            console.print(
                f"\nCreating table: '{settings.POINTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )
        conn.sql(create_table_sql)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Created table: '{settings.POINTS_TABLE_NAME}' in {time_elapsed:.3f} ms"  # noqa: E501
            )
        return True

    def ensure_hnsw_index(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> bool:
        """"""

        sql_stmt = (
            "INSTALL vss;\n"
            + "LOAD vss;\n"
            + "INSTALL json;\n"
            + "LOAD json;\n"
            + "SET hnsw_enable_experimental_persistence = true;\n"  # Required for HNSW index  # noqa: E501
            + SQL_STMT_CREATE_EMBEDDING_INDEX.format(
                table_name=settings.POINTS_TABLE_NAME,
                column_name="embedding",
                metric="cosine",
            )
        ).strip()
        if debug:
            console.print(
                "\nCreating embedding hnsw index with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=sql_stmt)}\n"
            )
        conn.sql(sql_stmt)

        return True

    def retrieve(
        self,
        point_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
        with_embedding: bool = False,
    ) -> "Point":
        """"""

        time_start = time.perf_counter() if debug else None

        # Get columns
        columns = list(self.model.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {settings.POINTS_TABLE_NAME} WHERE point_id = ?"  # noqa: E501
        parameters = [point_id]
        if debug:
            console.print(
                f"\nRetrieving point: '{point_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()

        if result is None:
            raise NotFoundError(
                f"Point with ID '{point_id}' not found.",
                response=httpx.Response(status_code=404),
                body=None,
            )

        data = dict(zip([c for c in columns], result))
        out = self.model.model_validate(data)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Retrieved point: '{point_id}' in {time_elapsed:.3f} ms")
        return out

    def create(
        self,
        point: Union["Point", Dict],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> "Point":
        points = self.bulk_create(points=[point], conn=conn, debug=debug)
        return points[0]

    def bulk_create(
        self,
        points: Union[
            Sequence["Point"], Sequence[Dict], Sequence[Union["Point", Dict]]
        ],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> List["Point"]:
        """"""

        time_start = time.perf_counter() if debug else None

        if not points:
            return []
        points = [
            self.model.model_validate(p) if isinstance(p, Dict) else p for p in points
        ]
        for idx, pt in enumerate(points):
            if not pt.is_embedded:
                raise ValueError(
                    f"Points[{idx}] is not embedded, please embed it first."
                )

        # Get columns
        columns = list(points[0].model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: List[Tuple[Any, ...]] = []
        for pt in points:
            parameters.append(tuple([getattr(pt, c) for c in columns]))

        query = (
            f"INSERT INTO {settings.POINTS_TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )
        if debug:
            _display_params = display_sql_parameters(parameters)
            console.print(
                "\nCreating points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
            )

        # Create points
        conn.executemany(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = time_end - time_start
            if time_elapsed > 1.0:
                console.print(f"Created {len(points)} points in {time_elapsed:.3f} s")
            else:
                time_elapsed *= 1000
                console.print(f"Created {len(points)} points in {time_elapsed:.3f} ms")
        return points

    def update(self, *args, **kwargs):
        raise APIStatusError(
            "Updating points is not supported.",
            response=httpx.Response(status_code=501),
            body=None,
        )

    def remove(
        self,
        point_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> None:
        """"""

        time_start = time.perf_counter() if debug else None

        query = f"DELETE FROM {settings.POINTS_TABLE_NAME} WHERE point_id = ?"
        parameters = [point_id]
        if debug:
            console.print(
                f"\nDeleting point: '{point_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Deleted point: '{point_id}' in {time_elapsed:.3f} ms")
        return None

    def list(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        after: Optional[Text] = None,
        before: Optional[Text] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "asc",
        conn: "duckdb.DuckDBPyConnection",
        with_embedding: bool = False,
        debug: bool = False,
    ) -> Pagination["Point"]:
        """"""

        time_start = time.perf_counter() if debug else None

        columns = list(self.model.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {settings.POINTS_TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if document_id is not None:
            where_clauses.append("document_id = ?")
            parameters.append(document_id)
        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

        if after is not None and order == "asc":
            where_clauses.append("point_id > ?")
            parameters.append(after)
        elif before is not None and order == "desc":
            where_clauses.append("point_id < ?")
            parameters.append(before)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        query += f"ORDER BY point_id {order.upper()}\n"

        fetch_limit = limit + 1
        query += f"LIMIT {fetch_limit}"

        if debug:
            console.print(
                "\nListing points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        results: List[Dict] = [
            {
                column: json.loads(value) if column == "metadata" else value
                for column, value in zip(columns, row)
            }
            for row in conn.execute(query, parameters).fetchall()
        ]

        points = [self.model.model_validate(row) for row in results[:limit]]

        out = Pagination.model_validate(
            {
                "data": points,
                "object": "list",
                "first_id": points[0].point_id if points else None,
                "last_id": points[-1].point_id if points else None,
                "has_more": len(results) > limit,
            }
        )

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Listed points in {time_elapsed:.3f} ms")
        return out

    def gen(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        after: Optional[Text] = None,
        before: Optional[Text] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "asc",
        conn: "duckdb.DuckDBPyConnection",
        with_embedding: bool = False,
        debug: bool = False,
    ) -> Generator["Point", None, None]:
        """"""  # noqa: E501

        has_more = True
        after = None
        while has_more:
            points = self.list(
                document_id=document_id,
                content_md5=content_md5,
                after=after,
                before=before,
                limit=limit,
                order=order,
                conn=conn,
                with_embedding=with_embedding,
                debug=debug,
            )
            has_more = points.has_more
            after = points.last_id
            for pt in points.data:
                yield pt
        return None

    def count(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> int:
        """"""

        time_start = time.perf_counter() if debug else None

        query = f"SELECT COUNT(*) FROM {settings.POINTS_TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if document_id is not None:
            where_clauses.append("document_id = ?")
            parameters.append(document_id)
        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        if debug:
            console.print(
                "\nCounting points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()
        count = result[0] if result else 0

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Counted points in {time_elapsed:.3f} ms")
        return count

    def drop(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        force: bool = False,
        debug: bool = False,
    ) -> None:
        """"""

        if not force:
            raise ValueError("Use force=True to drop table.")

        time_start = time.perf_counter() if debug else None

        query_template = jinja2.Template(SQL_STMT_DROP_TABLE)
        query = query_template.render(table_name=settings.POINTS_TABLE_NAME)
        if debug:
            console.print(
                f"\nDropping table: '{settings.POINTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
            )

        # Drop table
        conn.sql(query)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Dropped table: '{settings.POINTS_TABLE_NAME}' "
                + f"in {time_elapsed:.3f} ms"
            )
        return None

    def remove_outdated(
        self,
        *,
        document_id: Text,
        content_md5: Text,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> None:
        """"""

        time_start = time.perf_counter() if debug else None

        query_template = jinja2.Template(SQL_STMT_REMOVE_OUTDATED_POINTS)
        query = query_template.render(table_name=settings.POINTS_TABLE_NAME)
        parameters = [document_id, content_md5]

        if debug:
            console.print(
                "\nRemoving outdated points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        # Remove outdated points
        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Deleted outdated points of document: '{document_id}' in "
                + f"{time_elapsed:.3f} ms"
            )
        return None

    def remove_many(
        self,
        point_ids: Optional[List[Text]] = None,
        *,
        document_ids: Optional[List[Text]] = None,
        content_md5s: Optional[List[Text]] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> None:
        """"""

        if not any([point_ids, document_ids, content_md5s]):
            return None

        time_start = time.perf_counter() if debug else None

        query = f"DELETE FROM {settings.POINTS_TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if point_ids is not None:
            _placeholders = ", ".join(["?" for _ in point_ids])
            where_clauses.append(f"point_id IN ( {_placeholders} )")
            parameters.extend(point_ids)
        if document_ids is not None:
            _placeholders = ", ".join(["?" for _ in document_ids])
            where_clauses.append(f"document_id IN ( {_placeholders} )")
            parameters.extend(document_ids)
        if content_md5s is not None:
            _placeholders = ", ".join(["?" for _ in content_md5s])
            where_clauses.append(f"content_md5 IN ( {_placeholders} )")
            parameters.extend(content_md5s)

        if where_clauses:
            query += "WHERE " + " OR ".join(where_clauses) + "\n"

        if debug:
            console.print(
                "\nRemoving points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Deleted points in {time_elapsed:.3f} ms")
        return None


class DocumentQuerySet:
    def __init__(self, model: Type["Document"], *args, **kwargs):
        self.model = model
        self.__args = args
        self.__kwargs = kwargs

    def touch(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        raise_if_exists: bool = False,
        debug: bool = False,
    ) -> bool:
        """"""

        time_start = time.perf_counter() if debug else None

        # Check if table exists
        if (
            settings.DOCUMENTS_TABLE_NAME in show_tables(conn=conn)
            and raise_if_exists is True
        ):
            raise ConflictError(
                f"Table '{settings.DOCUMENTS_TABLE_NAME}' already exists.",
                response=httpx.Response(status_code=409),
                body=None,
            )

        # Install JSON and VSS extensions
        install_extensions(conn=conn, debug=debug)

        # Create table
        create_table_sql = openapi_to_create_table_sql(
            self.model.model_json_schema(),
            table_name=settings.DOCUMENTS_TABLE_NAME,
            primary_key="document_id",
            unique_fields=[],
            # unique_fields=["name"],  # Index limitations (https://duckdb.org/docs/sql/indexes)  # noqa: E501
            indexes=["content_md5"],
        )
        if debug:
            console.print(
                f"\nCreating table: '{settings.DOCUMENTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )

        conn.sql(create_table_sql)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Created table: '{settings.DOCUMENTS_TABLE_NAME}' in "
                + f"{time_elapsed:.3f} ms"
            )
        return True

    def retrieve(
        self,
        document_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        with_embedding: bool = False,
        debug: bool = False,
    ) -> Optional["Document"]:
        """"""

        time_start = time.perf_counter() if debug else None

        columns = list(self.model.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = (
            f"SELECT {columns_expr} FROM {settings.DOCUMENTS_TABLE_NAME} "
            + "WHERE document_id = ?"
        )
        parameters = [document_id]
        if debug:
            console.print(
                f"\nRetrieving document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()

        if result is None:
            raise NotFoundError(
                f"Document with ID '{document_id}' not found.",
                response=httpx.Response(status_code=404),
                body=None,
            )

        data = dict(zip(columns, result))
        data["metadata"] = json.loads(data["metadata"])
        out = self.model.model_validate(data)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Retrieved document: '{document_id}' in {time_elapsed:.3f} ms"
            )
        return out

    def create(
        self,
        document: Union["Document", Dict],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> "Document":
        """"""

        docs = self.bulk_create([document], conn=conn, debug=debug)
        return docs[0]

    def bulk_create(
        self,
        documents: Union[
            Sequence["Document"], Sequence[Dict], Sequence[Union["Document", Dict]]
        ],
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> List["Document"]:
        """"""

        time_start = time.perf_counter() if debug else None

        documents = [
            (
                self.model.model_validate(doc).strip()
                if isinstance(doc, Dict)
                else doc.strip()
            )
            for doc in documents
        ]

        columns = list(documents[0].model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: List[Tuple[Any, ...]] = [
            tuple(getattr(doc, c) for c in columns) for doc in documents
        ]

        query = (
            f"INSERT INTO {settings.DOCUMENTS_TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )
        if debug:
            _display_params = display_sql_parameters(parameters)
            console.print(
                "\nCreating documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
            )

        # Create documents
        conn.executemany(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = time_end - time_start
            if time_elapsed > 1.0:
                console.print(
                    f"Created {len(documents)} documents in {time_elapsed:.3f} s"
                )
            else:
                time_elapsed *= 1000
                console.print(
                    f"Created {len(documents)} documents in {time_elapsed:.3f} ms"
                )
        return documents

    def update(
        self,
        document_id: Text,
        *,
        name: Optional[Text] = None,
        content: Optional[Text] = None,
        metadata: Optional[Dict] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> "Document":
        """"""

        if not any([name, content, metadata]):
            raise ValueError("At least one of the parameters must be provided.")

        time_start = time.perf_counter() if debug else None

        # Check if the new name already exists
        if name is not None:
            existing_doc = conn.execute(
                "SELECT document_id FROM documents WHERE name = ? AND document_id != ?",
                [name, document_id],
            ).fetchone()
            if existing_doc:
                raise ConflictError(
                    f"The name '{name}' is already used by another document.",
                    response=httpx.Response(status_code=409),
                    body=None,
                )

        document = self.retrieve(document_id, conn=conn)
        if document is None:
            raise NotFoundError(
                f"Document with ID '{document_id}' not found.",
                response=httpx.Response(status_code=404),
                body=None,
            )

        set_query: List[Text] = []
        parameters = []
        if name is not None:
            document.name = name
            set_query.append("name = ?")
            parameters.append(document.name)
        if content is not None:
            document.content = content
            document.strip()
            set_query.append("content = ?")
            parameters.append(document.content)
        if metadata is not None:
            document.metadata = {} if document.metadata is None else document.metadata
            document.metadata.update(metadata)
            set_query.append("metadata = json_merge_patch(metadata, ?::JSON)")
            parameters.append(json.dumps(metadata))
        document.updated_at = int(time.time())
        set_query.append("updated_at = ?")
        parameters.append(document.updated_at)

        set_query_expr = ",\n    ".join(set_query)
        parameters.append(document_id)
        query = f"UPDATE {settings.DOCUMENTS_TABLE_NAME}\n"
        query += f"SET {set_query_expr}\n"
        query += "WHERE document_id = ?"
        if debug:
            console.print(
                f"\nUpdating document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Updated document: '{document_id}' in {time_elapsed:.3f} ms")
        return document

    def remove(
        self,
        document_id: Text,
        *,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> None:
        """"""

        time_start = time.perf_counter() if debug else None

        # Prepare delete query
        query = f"DELETE FROM {settings.DOCUMENTS_TABLE_NAME} WHERE document_id = ?"
        parameters = [document_id]
        if debug:
            console.print(
                f"\nDeleting document: '{document_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        # Delete document
        conn.execute(query, parameters)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Deleted document: '{document_id}' in {time_elapsed:.3f} ms")
        return None

    def list(
        self,
        *,
        after: Optional[Text] = None,
        before: Optional[Text] = None,
        limit: int = 20,
        order: Literal["asc", "desc"] = "asc",
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> Pagination["Document"]:
        """"""

        time_start = time.perf_counter() if debug else None

        columns = list(self.model.model_json_schema()["properties"].keys())
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {settings.DOCUMENTS_TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

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

        if debug:
            console.print(
                "\nListing documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        results = conn.execute(query, parameters).fetchall()
        results = [
            {
                column: json.loads(value) if column == "metadata" else value
                for column, value in zip(columns, row)
            }
            for row in results
        ]

        documents = [self.model.model_validate(row) for row in results[:limit]]

        out = Pagination.model_validate(
            {
                "data": documents,
                "object": "list",
                "first_id": documents[0].document_id if documents else None,
                "last_id": documents[-1].document_id if documents else None,
                "has_more": len(results) > limit,
            }
        )

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Listed documents in {time_elapsed:.3f} ms")
        return out

    def count(
        self,
        *,
        document_id: Optional[Text] = None,
        content_md5: Optional[Text] = None,
        conn: "duckdb.DuckDBPyConnection",
        debug: bool = False,
    ) -> int:
        """"""

        time_start = time.perf_counter() if debug else None

        query = f"SELECT COUNT(*) FROM {settings.DOCUMENTS_TABLE_NAME}\n"
        where_clauses: List[Text] = []
        parameters: List[Text] = []

        if document_id is not None:
            where_clauses.append("document_id = ?")
            parameters.append(document_id)
        if content_md5 is not None:
            where_clauses.append("content_md5 = ?")
            parameters.append(content_md5)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        if debug:
            console.print(
                "\nCounting documents with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = conn.execute(query, parameters).fetchone()
        count = result[0] if result else 0

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(f"Counted documents in {time_elapsed:.3f} ms")
        return count

    def drop(
        self,
        *,
        conn: "duckdb.DuckDBPyConnection",
        force: bool = False,
        debug: bool = False,
    ) -> None:
        """"""

        if not force:
            raise ValueError("Use force=True to drop table.")

        time_start = time.perf_counter() if debug else None

        query_template = jinja2.Template(SQL_STMT_DROP_TABLE)
        query = query_template.render(table_name=settings.DOCUMENTS_TABLE_NAME)
        if debug:
            console.print(
                f"\nDropping table: '{settings.DOCUMENTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
            )

        # Drop table
        conn.sql(query)

        if time_start is not None:
            time_end = time.perf_counter()
            time_elapsed = (time_end - time_start) * 1000
            console.print(
                f"Dropped table: '{settings.DOCUMENTS_TABLE_NAME}' "
                + f"in {time_elapsed:.3f} ms"
            )
        return None


class PointQuerySetDescriptor:
    def __get__(self, instance: None, owner: Type["Point"]) -> "PointQuerySet":
        if instance is not None:
            raise AttributeError(
                "PointQuerySetDescriptor cannot be accessed via an instance."
            )
        return PointQuerySet(owner)


class DocumentQuerySetDescriptor:
    def __get__(self, instance: None, owner: Type["Document"]) -> "DocumentQuerySet":
        if instance is not None:
            raise AttributeError(
                "DocumentQuerySetDescriptor cannot be accessed via an instance."
            )
        return DocumentQuerySet(owner)
