# dvs/db/graph/nodes/api.py
import logging
import typing

import duckdb

import dvs
from dvs.types.node import Node as NodeType
from dvs.types.paginations import Pagination
from dvs.utils.debug_print import debug_print
from dvs.utils.openapi import openapi_to_create_table_sql
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Nodes:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

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
        raise NotImplementedError

    def create(
        self,
        node: typing.Union[NodeType, typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> NodeType:
        raise NotImplementedError

    def bulk_create(
        self, nodes: typing.Sequence[NodeType], *, verbose: bool | None = None
    ) -> typing.List[NodeType]:
        raise NotImplementedError

    def list(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[NodeType]:
        raise NotImplementedError

    def gen(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[NodeType, None, None]:
        raise NotImplementedError

    def count(self, *, verbose: bool | None = None) -> int:
        raise NotImplementedError
