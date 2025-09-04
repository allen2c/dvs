# dvs/db/graph/edges/api.py
import functools
import logging
import typing

import duckdb
import openai

import dvs
from dvs.types.edge import Edge as EdgeType
from dvs.types.edge import RelationType
from dvs.types.paginations import Pagination
from dvs.utils.debug_print import debug_print
from dvs.utils.display import DISPLAY_SQL_PARAMS, display_sql_parameters
from dvs.utils.dummies import dummy_httpx_response
from dvs.utils.openapi import openapi_to_create_table_sql
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Edges:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    @functools.cached_property
    def columns(self) -> typing.Tuple[typing.Text, ...]:
        return tuple(EdgeType.model_json_schema()["properties"].keys())

    @functools.cached_property
    def columns_expr(self) -> typing.Text:
        return ", ".join(self.columns)

    def touch(self, *, verbose: bool | None = None) -> bool:
        with Timer() as timer:
            create_table_sql = openapi_to_create_table_sql(
                EdgeType.model_json_schema(),
                table_name=dvs.DVS_EDGES_TABLE_NAME,
                primary_key="edge_id",
                unique_fields=[],
                indexes=["edge_id", "relation", "from_node", "to_node"],
            )
            try:
                self.dvs.conn.sql(create_table_sql)
            except duckdb.CatalogException as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Table '{dvs.DVS_EDGES_TABLE_NAME}' already exists")
                else:
                    raise e

        debug_print(
            create_table_sql,
            title=f"Creating table: '{dvs.DVS_EDGES_TABLE_NAME}' with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        return True

    def retrieve(
        self, edge_id: typing.Text, *, verbose: bool | None = None
    ) -> EdgeType:
        query = f"SELECT {self.columns_expr} FROM {dvs.DVS_EDGES_TABLE_NAME}"

        with Timer() as timer:
            result = self.dvs.conn.execute(query).fetchone()

        debug_print(
            f"{query}",
            title="Retrieving edge with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        if result is None:
            raise openai.NotFoundError(
                f"Edge with ID '{edge_id}' not found.",
                response=dummy_httpx_response(404, b"Not Found"),
                body=None,
            )

        data = dict(zip(self.columns, result))
        edge = EdgeType.model_validate(data)

        return edge

    def create(
        self,
        edge: typing.Union[EdgeType, typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> EdgeType:
        edges = self.bulk_create(
            [edge if isinstance(edge, EdgeType) else EdgeType.model_validate(edge)],
            verbose=verbose,
        )
        return edges[0]

    def bulk_create(
        self, edges: typing.Sequence[EdgeType], *, verbose: bool | None = None
    ) -> typing.List[EdgeType]:
        if not edges:
            return []

        placeholders = ", ".join(["?" for _ in self.columns])
        parameters: typing.List[typing.Tuple[typing.Any, ...]] = [
            tuple(getattr(edge, c) for c in self.columns) for edge in edges
        ]

        query = (
            f"INSERT INTO {dvs.DVS_EDGES_TABLE_NAME} ({self.columns_expr}) "
            + f"VALUES ({placeholders})"
        )

        # Create nodes
        with Timer() as timer:
            logger.debug(f"🔨 Creating {len(edges)} edges ...")
            self.dvs.conn.executemany(query, parameters)

        debug_print(
            f"{query}\n{DISPLAY_SQL_PARAMS.format(params=display_sql_parameters(parameters))}",  # noqa: E501
            title="Creating edges with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )
        logger.info(f"✅ Created {len(edges)} edges.")
        return list(edges)

    def list(
        self,
        *,
        relation: typing.Optional[RelationType] = None,
        from_node_label_contains: typing.Optional[typing.Text] = None,
        to_node_label_contains: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[EdgeType]:
        query = f"SELECT {self.columns_expr} FROM {dvs.DVS_EDGES_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

        # Find edges with relation.
        # Or from_node_label, and to_node_label containing the substring
        # with insensitive case
        if relation is not None:
            where_clauses.append("relation = ?")
            parameters.append(relation)
        if from_node_label_contains is not None:
            where_clauses.append("from_node ILIKE ?")
            parameters.append(f"%{from_node_label_contains}%")
        if to_node_label_contains is not None:
            where_clauses.append("to_node ILIKE ?")
            parameters.append(f"%{to_node_label_contains}%")

        if after is not None:
            if order == "asc":
                where_clauses.append("edge_id > ?")
                parameters.append(after)
            elif order == "desc":
                where_clauses.append("edge_id < ?")
                parameters.append(after)
        elif before is not None:
            if order == "asc":
                where_clauses.append("edge_id < ?")
                parameters.append(before)
            elif order == "desc":
                where_clauses.append("edge_id > ?")
                parameters.append(before)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        query += f"ORDER BY edge_id {order.upper()}\n"

        # Fetch one more than the limit to determine if there are more results
        fetch_limit = limit + 1
        query += f"LIMIT {fetch_limit}"

        with Timer() as timer:
            results = self.dvs.conn.execute(query, parameters).fetchall()

        debug_print(
            f"{query}\n{DISPLAY_SQL_PARAMS.format(params=parameters)}",
            title="Listing edges with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        results = [
            {column: value for column, value in zip(self.columns, row)}
            for row in results
        ]

        edges = [EdgeType.model_validate(row) for row in results[:limit]]

        out = Pagination.model_validate(
            {
                "data": edges,
                "object": "list",
                "first_id": edges[0].edge_id if edges else None,
                "last_id": edges[-1].edge_id if edges else None,
                "has_more": len(results) > limit,
            }
        )

        return out

    def gen(
        self,
        *,
        relation: typing.Optional[RelationType] = None,
        from_node_label_contains: typing.Optional[typing.Text] = None,
        to_node_label_contains: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[EdgeType, None, None]:
        has_more = True
        current_after = after
        while has_more:
            edges = self.list(
                relation=relation,
                from_node_label_contains=from_node_label_contains,
                to_node_label_contains=to_node_label_contains,
                after=current_after,
                before=before,
                limit=limit,
                order=order,
                verbose=verbose,
            )
            has_more = edges.has_more
            current_after = edges.last_id
            for edge in edges.data:
                yield edge

    def count(
        self,
        *,
        relation: typing.Optional[RelationType] = None,
        from_node_label_contains: typing.Optional[typing.Text] = None,
        to_node_label_contains: typing.Optional[typing.Text] = None,
        verbose: bool | None = None,
    ) -> int:
        query = f"SELECT COUNT(*) FROM {dvs.DVS_EDGES_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

        if relation is not None:
            where_clauses.append("relation = ?")
            parameters.append(relation)
        if from_node_label_contains is not None:
            where_clauses.append("from_node ILIKE ?")
            parameters.append(f"%{from_node_label_contains}%")
        if to_node_label_contains is not None:
            where_clauses.append("to_node ILIKE ?")
            parameters.append(f"%{to_node_label_contains}%")

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        with Timer() as timer:
            result = self.dvs.conn.execute(query, parameters).fetchone()

        debug_print(
            f"{query}\n{DISPLAY_SQL_PARAMS.format(params=parameters)}",
            title="Counting edges with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        count = result[0] if result else 0

        return count
