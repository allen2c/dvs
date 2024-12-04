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
import httpx
import jinja2
from openai import APIStatusError, ConflictError, NotFoundError
from tqdm import tqdm

from dvs.config import console, settings
from dvs.types.paginations import Pagination
from dvs.utils.chunk import chunks
from dvs.utils.display import (
    DISPLAY_SQL_PARAMS,
    DISPLAY_SQL_QUERY,
    display_sql_parameters,
)
from dvs.utils.openapi import openapi_to_create_table_sql
from dvs.utils.sql_stmts import (
    SQL_STMT_CREATE_EMBEDDING_INDEX,
    SQL_STMT_DROP_TABLE,
    SQL_STMT_INSTALL_EXTENSIONS,
    SQL_STMT_REMOVE_OUTDATED_POINTS,
    SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE,
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
    if debug:
        console.print(
            "\nInstalling extensions with SQL:\n"
            + f"{DISPLAY_SQL_QUERY.format(sql=SQL_STMT_INSTALL_EXTENSIONS)}\n"
        )
    conn.sql(SQL_STMT_INSTALL_EXTENSIONS)


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
            SQL_STMT_INSTALL_EXTENSIONS
            + f"\n{create_table_sql}\n"
            + f"\n{SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE}\n"  # Required for HNSW index  # noqa: E501
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
        """
        Ensure HNSW (Hierarchical Navigable Small World) index exists for vector similarity search.

        Creates or updates the HNSW index on the embedding column to enable efficient vector
        similarity searches. This function installs necessary DuckDB extensions, enables
        experimental HNSW persistence, and creates the index using cosine similarity metric.

        Notes
        -----
        - HNSW indexing is crucial for performant vector similarity searches
        - The index is created on the 'embedding' column of the points table
        - Uses cosine similarity as the distance metric
        - Enables experimental HNSW persistence for index durability

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Point.objects.ensure_hnsw_index(conn=conn, debug=True)
        Creating embedding hnsw index with SQL:
        ...
        True

        Warnings
        --------
        This operation may take significant time on large datasets as it needs to build
        the HNSW index structure.
        """  # noqa: E501

        sql_stmt = (
            SQL_STMT_INSTALL_EXTENSIONS
            + f"\n{SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE}\n"  # Required for HNSW index  # noqa: E501
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
        """
        Retrieve a single point from the DuckDB database by its ID.

        This method fetches a point record from the database, optionally including its embedding
        vector, and validates the data against the Point model schema. If the point is not found,
        raises a NotFoundError.

        Notes
        -----
        - When with_embedding=False, the embedding vector is excluded from the query results
        to reduce data transfer
        - Debug mode provides SQL query details and timing information
        - The point's metadata is automatically parsed from JSON format

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> point = Point.objects.retrieve(
        ...     point_id='pt_123',
        ...     conn=conn,
        ...     debug=True
        ... )
        Retrieving point: 'pt_123' with SQL:
        ...
        Retrieved point: 'pt_123' in 1.234 ms

        Warnings
        --------
        Raises NotFoundError with status code 404 if the point_id doesn't exist in the database.
        """  # noqa: E501

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
        """
        Create a single point in the DuckDB database.

        A convenience method that wraps bulk_create for single point insertion. The point
        must be properly embedded before creation.

        Notes
        -----
        - The point's embedding vector must be set before calling this method
        - Debug mode provides SQL query details and timing information
        - The point's metadata is automatically handled as JSON in the database

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> point = Point(point_id='pt_123', embedding=[0.1, 0.2, ...])
        >>> created_point = Point.objects.create(point=point, conn=conn, debug=True)
        Creating points with SQL:
        ...
        Created 1 points in 1.234 ms

        Warnings
        --------
        Raises ValueError if the point is not embedded before creation.
        """  # noqa: E501

        points = self.bulk_create(points=[point], conn=conn, debug=debug)
        return points[0]

    def bulk_create(
        self,
        points: Union[
            Sequence["Point"], Sequence[Dict], Sequence[Union["Point", Dict]]
        ],
        *,
        conn: "duckdb.DuckDBPyConnection",
        batch_size: int = 20,
        debug: bool = False,
    ) -> List["Point"]:
        """
        Create multiple points in the DuckDB database in batches.

        This method efficiently inserts multiple Point records into the database, validating
        each point and ensuring they have proper embeddings. Points are inserted in batches
        to optimize performance.

        Notes
        -----
        - Points must be embedded before creation (have valid embedding vectors)
        - Points can be provided as Point objects or dictionaries
        - Batch processing helps manage memory usage for large datasets
        - Debug mode provides SQL query details and progress bar

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> points = [Point(point_id='pt_1', embedding=[...]), Point(point_id='pt_2', embedding=[...])]
        >>> created_points = Point.objects.bulk_create(
        ...     points=points,
        ...     conn=conn,
        ...     batch_size=100,
        ...     debug=True
        ... )
        Creating points with SQL:
        ...
        Created 2 points in 1.234 ms

        Warnings
        --------
        Raises ValueError if any point in the sequence is not properly embedded.
        """  # noqa: E501

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

        # Paginate points creation
        _iter_batch_pts = (
            tqdm(
                chunks(points, batch_size=batch_size),
                total=len(points) // batch_size
                + (1 if len(points) % batch_size else 0),
                desc="Creating points",
            )
            if debug
            else chunks(points, batch_size=batch_size)
        )
        _shown_debug = False
        for _batch_pts in _iter_batch_pts:
            parameters: List[Tuple[Any, ...]] = []
            for pt in _batch_pts:
                parameters.append(tuple([getattr(pt, c) for c in columns]))

            query = (
                f"INSERT INTO {settings.POINTS_TABLE_NAME} ({columns_expr}) "
                + f"VALUES ({placeholders})"
            )
            query = SQL_STMT_INSTALL_EXTENSIONS + f"\n{query}\n"
            if debug and not _shown_debug:
                _display_params = display_sql_parameters(parameters)
                console.print(
                    "\nCreating points with SQL:\n"
                    + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                    + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
                )
                _shown_debug = True

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
        """
        Explicitly disallow updating points in the database.

        Notes
        -----
        This method is intentionally disabled to maintain data integrity and immutability
        of points in the vector database. Any attempt to update points will raise an
        APIStatusError with status code 501 (Not Implemented).

        Warnings
        --------
        Always raises APIStatusError when called, as point updates are not supported
        by design.
        """  # noqa: E501

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
        """
        Delete a single point from the DuckDB database by its ID.

        Notes
        -----
        - Executes a DELETE SQL statement targeting a specific point_id
        - Debug mode provides SQL query details and timing information
        - No error is raised if the point doesn't exist

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Point.objects.remove(
        ...     point_id='pt_123',
        ...     conn=conn,
        ...     debug=True
        ... )
        Deleting point: 'pt_123' with SQL:
        ...
        Deleted point: 'pt_123' in 1.234 ms
        """  # noqa: E501

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
        """
        List and paginate points from the DuckDB database with optional filtering.

        Notes
        -----
        - Supports filtering by document_id and content_md5
        - Implements cursor-based pagination using point_id
        - Can exclude embedding vectors to reduce response size
        - Orders results by point_id in ascending or descending order
        - Debug mode provides SQL query details and timing information

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> points = Point.objects.list(
        ...     document_id='doc_123',
        ...     limit=10,
        ...     order='asc',
        ...     conn=conn,
        ...     debug=True
        ... )
        Listing points with SQL:
        ...
        Listed points in 1.234 ms

        Warnings
        --------
        Including embeddings (with_embedding=True) can significantly increase response size
        and processing time for large result sets.
        """  # noqa: E501

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
        """
        Generate and yield points from the DuckDB database with pagination support.

        A generator wrapper around the list() method that handles pagination automatically,
        yielding individual points until all matching records have been retrieved. This is
        useful for processing large result sets without loading all points into memory at once.

        Notes
        -----
        - Automatically handles pagination using cursor-based pagination with point_id
        - Memory efficient as it yields points one at a time
        - Maintains the same filtering and ordering capabilities as the list() method

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> for point in Point.objects.gen(
        ...     document_id='doc_123',
        ...     limit=100,
        ...     conn=conn,
        ...     debug=True
        ... ):
        ...     process_point(point)
        """  # noqa: E501

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
        """
        Count points in the DuckDB database with optional filtering by document_id and content_md5.

        Notes
        -----
        - Executes a COUNT SQL query on the points table
        - Supports filtering by document_id and/or content_md5
        - Debug mode provides SQL query details and timing information

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> count = Point.objects.count(
        ...     document_id='doc_123',
        ...     conn=conn,
        ...     debug=True
        ... )
        Counting points with SQL:
        ...
        Counted points in 1.234 ms
        """  # noqa: E501

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
        """
        Drop the points table from the DuckDB database.

        Notes
        -----
        - Requires explicit force=True parameter as a safety measure
        - Debug mode provides SQL query details and timing information
        - Drops the table and all associated indexes/constraints

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Point.objects.drop(
        ...     conn=conn,
        ...     force=True,
        ...     debug=True
        ... )
        Dropping table: 'points' with SQL:
        ...
        Dropped table: 'points' in 1.234 ms

        Warnings
        --------
        This operation is irreversible and will permanently delete all points data.
        Use with caution.
        """

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
        """
        Remove outdated points associated with a document based on content hash.

        This method deletes all points belonging to a specific document that don't match
        the provided content_md5 hash, effectively cleaning up outdated vector embeddings
        when document content changes.

        Notes
        -----
        - Executes a DELETE SQL statement targeting points with matching document_id
        but different content_md5
        - Debug mode provides SQL query details and timing information
        - No error is raised if no points are deleted

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Point.objects.remove_outdated(
        ...     document_id='doc_123',
        ...     content_md5='abc123',
        ...     conn=conn,
        ...     debug=True
        ... )
        Removing outdated points with SQL:
        ...
        Deleted outdated points of document: 'doc_123' in 1.234 ms
        """  # noqa: E501

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
        """
        Delete multiple points from the database based on point IDs, document IDs, or content hashes.

        Notes
        -----
        - Accepts lists of point_ids, document_ids, or content_md5s for bulk deletion
        - Uses OR conditions between different identifier types (any match will be deleted)
        - Debug mode provides SQL query details and timing information
        - No error is raised if points don't exist

        Examples
        --------
        >>> conn = duckdb.connect('database.duckdb')
        >>> Point.objects.remove_many(
        ...     point_ids=['pt_1', 'pt_2'],
        ...     document_ids=['doc_1'],
        ...     conn=conn,
        ...     debug=True
        ... )
        Removing points with SQL:
        ...
        Deleted points in 1.234 ms
        """  # noqa: E501

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
        ...     with_embedding=True,
        ...     debug=True
        ... )
        """

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
        """
        Update a document in the DuckDB database with new values for name, content, or metadata.

        This function updates the specified document's fields in the database. It ensures that at least
        one of the parameters (name, content, metadata) is provided for the update. If a new name is
        provided, it checks for uniqueness across other documents. The function also updates the
        `updated_at` timestamp to the current time.

        Raises
        ------
        ValueError
            If none of the parameters (name, content, metadata) are provided.
        ConflictError
            If the new name is already used by another document.
        NotFoundError
            If the document with the specified ID is not found.

        Notes
        -----
        - The function uses parameterized SQL queries to prevent SQL injection.
        - Debug mode provides SQL query details and timing information.
        """  # noqa: E501

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
