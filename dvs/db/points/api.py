# dvs/db/points/api.py
import json
import logging
import typing

import duckdb
import jinja2
import openai
from tqdm import tqdm

import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.paginations import Pagination
from dvs.types.point import Point as PointType
from dvs.utils.chunk import chunks
from dvs.utils.display import (
    DISPLAY_SQL_PARAMS,
    DISPLAY_SQL_QUERY,
    display_sql_parameters,
)
from dvs.utils.dummies import dummy_httpx_response
from dvs.utils.ensure import ensure_dict
from dvs.utils.sql_stmts import (
    SQL_STMT_CREATE_EMBEDDING_INDEX,
    SQL_STMT_DROP_TABLE,
    SQL_STMT_INSTALL_EXTENSIONS,
    SQL_STMT_REMOVE_OUTDATED_POINTS,
    SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE,
)
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Points:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        """
        Initialize the points table in DuckDB with vector similarity search support.
        Creates the table with proper schema, indexes, and HNSW indexing for embeddings.
        """
        with Timer() as timer:
            self._touch(verbose=verbose)
        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Created table: '{dvs.POINTS_TABLE_NAME}' in {dur:.3f} ms")
        return True

    def retrieve(
        self,
        point_id: typing.Text,
        *,
        verbose: bool | None = None,
        with_embedding: bool = False,
    ) -> "PointType":
        """
        Retrieve a single point from the database by its ID.
        Raises NotFoundError if the point doesn't exist.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            out = self._retrieve(
                point_id, verbose=verbose, with_embedding=with_embedding
            )

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Retrieved point: '{point_id}' in {dur:.3f} ms")
        return out

    def create(
        self,
        point: typing.Union["PointType", typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> "PointType":
        """
        Create a single point in the database.
        The point must have a valid embedding vector before creation.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        point = (
            PointType.model_validate(point) if isinstance(point, typing.Dict) else point
        )

        with Timer() as timer:
            points = self._bulk_create(points=[point], verbose=verbose, batch_size=1)
        point = points[0]

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Created point: '{point.point_id}' in {dur:.3f} ms")
        return point

    def bulk_create(
        self,
        points: typing.Union[
            typing.Sequence["PointType"],
            typing.Sequence[typing.Dict],
            typing.Sequence[typing.Union["PointType", typing.Dict]],
        ],
        *,
        batch_size: int = 100,
        verbose: bool | None = None,
    ) -> typing.List["PointType"]:
        """
        Create multiple points in the database efficiently using batches.
        All points must have valid embedding vectors before creation.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        points = [
            PointType.model_validate(p) if isinstance(p, typing.Dict) else p
            for p in points
        ]

        with Timer() as timer:
            points = self._bulk_create(
                points=points, verbose=verbose, batch_size=batch_size
            )

        if verbose:
            dur, unit = (
                (timer.duration, "s")
                if timer.duration > 1.0
                else (timer.duration * 1000, "ms")
            )
            logger.debug(f"Created {len(points)} points in {dur:.3f} {unit}")

        return points

    def remove(self, point_id: typing.Text, *, verbose: bool | None = None) -> None:
        """
        Delete a single point from the database by its ID.
        No error is raised if the point doesn't exist.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            self._remove(point_id, verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Deleted point: '{point_id}' in {dur:.3f} ms")

        return None

    def list(
        self,
        *,
        document_id: typing.Optional[typing.Text] = None,
        content_md5: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> Pagination["PointType"]:
        """
        List and paginate points with optional filtering by document_id and content_md5.
        Uses cursor-based pagination with point_id for efficient large result sets.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            out = self._list(
                document_id=document_id,
                content_md5=content_md5,
                after=after,
                before=before,
                limit=limit,
                order=order,
                with_embedding=with_embedding,
                verbose=verbose,
            )

        if verbose:
            dur, unit = (
                (timer.duration, "s")
                if timer.duration > 1.0
                else (timer.duration * 1000, "ms")
            )
            logger.debug(f"Listed points in {dur:.3f} {unit}")
        return out

    def gen(
        self,
        *,
        document_id: typing.Optional[typing.Text] = None,
        content_md5: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> typing.Generator["PointType", None, None]:
        """
        Generate points with automatic pagination for memory-efficient processing.
        Yields individual points while handling pagination internally.
        """

        has_more = True
        after = None
        while has_more:
            points = self._list(
                document_id=document_id,
                content_md5=content_md5,
                after=after,
                before=before,
                limit=limit,
                order=order,
                with_embedding=with_embedding,
                verbose=verbose,
            )
            has_more = points.has_more
            after = points.last_id
            for pt in points.data:
                yield pt
        return None

    def count(
        self,
        *,
        document_id: typing.Optional[typing.Text] = None,
        content_md5: typing.Optional[typing.Text] = None,
        verbose: bool | None = None,
    ) -> int:
        """
        Count points in the database with optional filtering.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            count = self._count(
                document_id=document_id,
                content_md5=content_md5,
                verbose=verbose,
            )

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Counted points in {dur:.3f} ms")
        return count

    def drop(
        self,
        *,
        force: bool = False,
        touch_after_drop: bool = True,
        verbose: bool | None = None,
    ) -> None:
        """
        Drop the points table from the database.
        Requires force=True as a safety measure. This operation is irreversible.
        """

        if not force:
            raise ValueError("Use force=True to drop table.")

        with Timer() as timer:
            self._drop(verbose=verbose)
            if touch_after_drop:
                self._touch(verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Dropped points in {dur:.3f} ms")

        return None

    def remove_outdated(
        self,
        *,
        document_id: typing.Text,
        content_md5: typing.Text,
        verbose: bool | None = None,
    ) -> None:
        """
        Remove outdated points for a document that don't match the current content hash.
        Cleans up old vector embeddings when document content changes.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            self._remove_outdated(
                document_id=document_id,
                content_md5=content_md5,
                verbose=verbose,
            )

        if verbose:
            dur = timer.duration * 1000
            logger.debug(
                f"Deleted outdated points of document: '{document_id}' in {dur:.3f} ms"
            )
        return None

    def remove_many(
        self,
        point_ids: typing.Optional[typing.List[typing.Text]] = None,
        *,
        document_ids: typing.Optional[typing.List[typing.Text]] = None,
        content_md5s: typing.Optional[typing.List[typing.Text]] = None,
        verbose: bool | None = None,
    ) -> None:
        """
        Delete multiple points by point IDs, document IDs, or content hashes.
        Uses OR conditions between different identifier types.
        """
        verbose = self.dvs.verbose if verbose is None else verbose
        with Timer() as timer:
            self._remove_many(
                point_ids=point_ids,
                document_ids=document_ids,
                content_md5s=content_md5s,
                verbose=verbose,
            )

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Deleted points in {dur:.3f} ms")

        return None

    def _touch(self, *, verbose: bool | None = None) -> bool:
        """
        Initialize the points table with extensions and vector similarity search indexes.
        """  # noqa: E501
        # Install JSON and VSS extensions
        self.dvs.db.install_extensions(verbose=verbose)

        # Create table
        create_table_sql = openapi_utils.openapi_to_create_table_sql(
            PointType.model_json_schema(),
            table_name=dvs.POINTS_TABLE_NAME,
            primary_key="point_id",
            indexes=["document_id", "content_md5"],
            custom_sql_types={
                "embedding": f"FLOAT[{self.dvs.db_manifest.embedding_dimensions}]"
            },
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

        try:
            self.dvs.conn.sql(create_table_sql)
        except duckdb.CatalogException as e:
            if "already exists" in str(e).lower():
                logger.debug(f"Table '{dvs.POINTS_TABLE_NAME}' already exists")
            else:
                raise e

        return True

    def _retrieve(
        self,
        point_id: typing.Text,
        *,
        verbose: bool | None,
        with_embedding: bool,
    ) -> "PointType":
        """
        Retrieve a single point from the database by its ID.
        """
        # Get columns
        columns = list(PointType.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {dvs.POINTS_TABLE_NAME} WHERE point_id = ?"  # noqa: E501
        parameters = [point_id]
        if verbose:
            self.dvs.settings.console.print(
                f"\nRetrieving point: '{point_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = self.dvs.conn.execute(query, parameters).fetchone()

        if result is None:
            raise openai.NotFoundError(
                f"Point with ID '{point_id}' not found.",
                response=dummy_httpx_response(404, b"Not Found"),
                body=None,
            )

        data = dict(zip([c for c in columns], result))
        data["metadata"] = ensure_dict(data.get("metadata", None))
        out = PointType.model_validate(data)

        return out

    def _bulk_create(
        self,
        points: typing.Sequence["PointType"],
        *,
        verbose: bool | None,
        batch_size: int,
    ) -> typing.List["PointType"]:
        """
        Create multiple points in the database using batched inserts.
        """

        if not points:
            return []

        # Validate points
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
            if verbose
            else chunks(points, batch_size=batch_size)
        )
        _shown_debug = False
        for _batch_pts in _iter_batch_pts:
            parameters: typing.List[typing.Tuple[typing.Any, ...]] = []
            for pt in _batch_pts:
                parameters.append(
                    tuple(
                        [
                            getattr(pt, c) if c != "embedding" else pt.to_python()
                            for c in columns
                        ]
                    )
                )

            query = (
                f"INSERT INTO {dvs.POINTS_TABLE_NAME} ({columns_expr}) "
                + f"VALUES ({placeholders})"
            )
            query = SQL_STMT_INSTALL_EXTENSIONS + f"\n{query}\n"
            if verbose and not _shown_debug:
                _display_params = display_sql_parameters(parameters)
                self.dvs.settings.console.print(
                    "\nCreating points with SQL:\n"
                    + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                    + f"{DISPLAY_SQL_PARAMS.format(params=_display_params)}\n"
                )
                _shown_debug = True

            # Create points
            self.dvs.conn.executemany(query, parameters)

        return list(points)

    def _remove(self, point_id: typing.Text, *, verbose: bool | None) -> None:
        """
        Delete a single point from the database by its ID.
        """

        query = (
            SQL_STMT_INSTALL_EXTENSIONS
            + f"\nDELETE FROM {dvs.POINTS_TABLE_NAME} WHERE point_id = ?"
        )
        parameters = [point_id]
        if verbose:
            self.dvs.settings.console.print(
                f"\nDeleting point: '{point_id}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        self.dvs.conn.execute(query, parameters)

        return None

    def _list(
        self,
        *,
        document_id: typing.Optional[typing.Text],
        content_md5: typing.Optional[typing.Text],
        after: typing.Optional[typing.Text],
        before: typing.Optional[typing.Text],
        limit: int,
        order: typing.Literal["asc", "desc"],
        with_embedding: bool,
        verbose: bool | None,
    ) -> Pagination["PointType"]:
        """
        List and paginate points with optional filtering.
        """
        columns = list(PointType.model_json_schema()["properties"].keys())
        if not with_embedding:
            columns = [c for c in columns if c != "embedding"]
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {dvs.POINTS_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

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

        if verbose:
            self.dvs.settings.console.print(
                "\nListing points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        results: typing.List[typing.Dict] = [
            {
                column: json.loads(value) if column == "metadata" else value
                for column, value in zip(columns, row)
            }
            for row in self.dvs.conn.execute(query, parameters).fetchall()
        ]

        points = [PointType.model_validate(row) for row in results[:limit]]

        out = Pagination.model_validate(
            {
                "data": points,
                "object": "list",
                "first_id": points[0].point_id if points else None,
                "last_id": points[-1].point_id if points else None,
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
        Count points in the database with optional filtering.
        """
        query = f"SELECT COUNT(*) FROM {dvs.POINTS_TABLE_NAME}\n"
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
                "\nCounting points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        result = self.dvs.conn.execute(query, parameters).fetchone()
        count = result[0] if result else 0

        return count

    def _drop(self, *, verbose: bool | None) -> None:
        """
        Drop the points table from the database.
        """
        query_template = jinja2.Template(SQL_STMT_DROP_TABLE)
        query = query_template.render(table_name=dvs.POINTS_TABLE_NAME)
        if verbose:
            self.dvs.settings.console.print(
                f"\nDropping table: '{dvs.POINTS_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
            )

        # Drop table
        self.dvs.conn.sql(query)

        return None

    def _remove_outdated(
        self,
        *,
        document_id: typing.Text,
        content_md5: typing.Text,
        verbose: bool | None,
    ) -> None:
        """
        Remove outdated points for a document based on content hash.
        """
        query_template = jinja2.Template(SQL_STMT_REMOVE_OUTDATED_POINTS)
        query = query_template.render(table_name=dvs.POINTS_TABLE_NAME)
        parameters = [document_id, content_md5]

        if verbose:
            self.dvs.settings.console.print(
                "\nRemoving outdated points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        # Remove outdated points
        self.dvs.conn.execute(query, parameters)

        return None

    def _remove_many(
        self,
        point_ids: typing.Optional[typing.List[typing.Text]],
        *,
        document_ids: typing.Optional[typing.List[typing.Text]],
        content_md5s: typing.Optional[typing.List[typing.Text]],
        verbose: bool | None,
    ) -> None:
        """
        Delete multiple points by various identifiers using OR conditions.
        """

        if not any([point_ids, document_ids, content_md5s]):
            return None

        query = SQL_STMT_INSTALL_EXTENSIONS + f"\nDELETE FROM {dvs.POINTS_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

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

        if verbose:
            self.dvs.settings.console.print(
                "\nRemoving points with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        self.dvs.conn.execute(query, parameters)

        return None
