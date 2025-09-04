# dvs/db/graph/edges/api.py
import logging
import typing

import duckdb

import dvs
from dvs.types.edge import Edge as EdgeType
from dvs.types.paginations import Pagination
from dvs.utils.debug_print import debug_print
from dvs.utils.openapi import openapi_to_create_table_sql
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Edges:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

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
        raise NotImplementedError

    def create(
        self,
        edge: typing.Union[EdgeType, typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> EdgeType:
        raise NotImplementedError

    def bulk_create(
        self, edges: typing.Sequence[EdgeType], *, verbose: bool | None = None
    ) -> typing.List[EdgeType]:
        raise NotImplementedError

    def list(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[EdgeType]:
        raise NotImplementedError

    def gen(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[EdgeType, None, None]:
        raise NotImplementedError

    def count(self, *, verbose: bool | None = None) -> int:
        raise NotImplementedError
