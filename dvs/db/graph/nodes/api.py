# dvs/db/graph/nodes/api.py
import functools
import logging
import typing

import duckdb
import openai

import dvs
from dvs.types.node import Node as NodeType
from dvs.types.paginations import Pagination
from dvs.utils.debug_print import debug_print
from dvs.utils.display import DISPLAY_SQL_PARAMS, display_sql_parameters
from dvs.utils.dummies import dummy_httpx_response
from dvs.utils.openapi import openapi_to_create_table_sql
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Nodes:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    @functools.cached_property
    def columns(self) -> typing.Tuple[typing.Text, ...]:
        return tuple(NodeType.model_json_schema()["properties"].keys())

    @functools.cached_property
    def columns_expr(self) -> typing.Text:
        return ", ".join(self.columns)

    def touch(self, *, verbose: bool | None = None) -> bool:
        with Timer() as timer:
            create_table_sql = openapi_to_create_table_sql(
                NodeType.model_json_schema(),
                table_name=dvs.DVS_NODES_TABLE_NAME,
                primary_key="node_id",
                unique_fields=[],
                indexes=["node_id", "label"],
            )
            try:
                self.dvs.conn.sql(create_table_sql)
            except duckdb.CatalogException as e:
                if "already exists" in str(e).lower():
                    logger.debug(f"Table '{dvs.DVS_NODES_TABLE_NAME}' already exists")
                else:
                    raise e

        debug_print(
            create_table_sql,
            title=f"Creating table: '{dvs.DVS_NODES_TABLE_NAME}' with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        return True

    def retrieve(
        self, node_id: typing.Text, *, verbose: bool | None = None
    ) -> NodeType:
        query = f"SELECT {self.columns_expr} FROM {dvs.DVS_NODES_TABLE_NAME}"

        with Timer() as timer:
            result = self.dvs.conn.execute(query).fetchone()

        debug_print(
            f"{query}",
            title="Retrieving node with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        if result is None:
            raise openai.NotFoundError(
                f"Node with ID '{node_id}' not found.",
                response=dummy_httpx_response(404, b"Not Found"),
                body=None,
            )

        data = dict(zip(self.columns, result))
        node = NodeType.model_validate(data)

        return node

    def create(
        self,
        node: typing.Union[NodeType, typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> NodeType:
        nodes = self.bulk_create(
            [node if isinstance(node, NodeType) else NodeType.model_validate(node)],
            verbose=verbose,
        )
        return nodes[0]

    def bulk_create(
        self, nodes: typing.Sequence[NodeType], *, verbose: bool | None = None
    ) -> typing.List[NodeType]:
        if not nodes:
            return []

        placeholders = ", ".join(["?" for _ in self.columns])
        parameters: typing.List[typing.Tuple[typing.Any, ...]] = [
            tuple(getattr(node, c) for c in self.columns) for node in nodes
        ]

        query = (
            f"INSERT INTO {dvs.DVS_NODES_TABLE_NAME} ({self.columns_expr}) "
            + f"VALUES ({placeholders})"
        )

        # Create nodes
        with Timer() as timer:
            self.dvs.conn.executemany(query, parameters)

        debug_print(
            f"{query}\n{DISPLAY_SQL_PARAMS.format(params=display_sql_parameters(parameters))}",  # noqa: E501
            title="Creating nodes with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )
        return list(nodes)

    def list(
        self,
        *,
        label_contains: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[NodeType]:
        query = f"SELECT {self.columns_expr} FROM {dvs.DVS_NODES_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

        # Find nodes with label containing the substring with insensitive case
        if label_contains is not None:
            where_clauses.append("label ILIKE ?")
            parameters.append(f"%{label_contains}%")

        if after is not None:
            if order == "asc":
                where_clauses.append("node_id > ?")
                parameters.append(after)
            elif order == "desc":
                where_clauses.append("node_id < ?")
                parameters.append(after)
        elif before is not None:
            if order == "asc":
                where_clauses.append("node_id < ?")
                parameters.append(before)
            elif order == "desc":
                where_clauses.append("node_id > ?")
                parameters.append(before)

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        query += f"ORDER BY document_id {order.upper()}\n"

        # Fetch one more than the limit to determine if there are more results
        fetch_limit = limit + 1
        query += f"LIMIT {fetch_limit}"

        with Timer() as timer:
            results = self.dvs.conn.execute(query, parameters).fetchall()

        debug_print(
            f"{query}\n{DISPLAY_SQL_PARAMS.format(params=parameters)}",
            title="Listing documents with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        results = [
            {column: value for column, value in zip(self.columns, row)}
            for row in results
        ]

        nodes = [NodeType.model_validate(row) for row in results[:limit]]

        out = Pagination.model_validate(
            {
                "data": nodes,
                "object": "list",
                "first_id": nodes[0].node_id if nodes else None,
                "last_id": nodes[-1].node_id if nodes else None,
                "has_more": len(results) > limit,
            }
        )

        return out

    def gen(
        self,
        *,
        label_contains: typing.Optional[typing.Text] = None,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[NodeType, None, None]:
        has_more = True
        current_after = after
        while has_more:
            nodes = self.list(
                label_contains=label_contains,
                after=current_after,
                before=before,
                limit=limit,
                order=order,
                verbose=verbose,
            )
            has_more = nodes.has_more
            current_after = nodes.last_id
            for node in nodes.data:
                yield node

    def count(
        self,
        *,
        label_contains: typing.Optional[typing.Text] = None,
        verbose: bool | None = None,
    ) -> int:
        query = f"SELECT COUNT(*) FROM {dvs.DVS_NODES_TABLE_NAME}\n"
        where_clauses: typing.List[typing.Text] = []
        parameters: typing.List[typing.Text] = []

        if label_contains is not None:
            where_clauses.append("label ILIKE ?")
            parameters.append(f"%{label_contains}%")

        if where_clauses:
            query += "WHERE " + " AND ".join(where_clauses) + "\n"

        with Timer() as timer:
            result = self.dvs.conn.execute(query, parameters).fetchone()

        debug_print(
            f"{query}\n{DISPLAY_SQL_PARAMS.format(params=parameters)}",
            title="Counting nodes with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        count = result[0] if result else 0

        return count
