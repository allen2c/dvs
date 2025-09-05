import functools
import logging
import textwrap
import typing
from concurrent.futures import ThreadPoolExecutor

import dvs
from dvs.types.edge import Edge as EdgeType
from dvs.types.edge import (
    RelationHasA,
    RelationIsA,
    RelationIsFrom,
    RelationRelatedTo,
    RelationType,
)
from dvs.types.node import Node as NodeType
from dvs.utils.debug_print import debug_print
from dvs.utils.sql_stmts import SQL_STMT_LOAD_DUCKPGQ
from dvs.utils.timer import Timer

if typing.TYPE_CHECKING:
    from dvs.db.graph.edges.api import Edges
    from dvs.db.graph.nodes.api import Nodes

logger = logging.getLogger(__name__)


class Graph:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        self.nodes.touch(verbose=verbose)
        self.edges.touch(verbose=verbose)

        with Timer() as timer:
            conn = self.dvs.conn
            # First, execute the extension installation
            conn.execute("LOAD duckpgq;")

            # Then, create the property graph
            create_table_sql = textwrap.dedent(
                f"""
                CREATE PROPERTY GRAPH {dvs.DVS_GRAPH_TABLE_NAME}
                VERTEX TABLES (
                    {dvs.DVS_NODES_TABLE_NAME}
                )
                EDGE TABLES (
                    {dvs.DVS_EDGES_IS_A_TABLE_NAME}
                        SOURCE KEY (from_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        DESTINATION KEY (to_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        LABEL {RelationIsA},
                    {dvs.DVS_EDGES_HAS_A_TABLE_NAME}
                        SOURCE KEY (from_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        DESTINATION KEY (to_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        LABEL {RelationHasA},
                    {dvs.DVS_EDGES_RELATED_TO_TABLE_NAME}
                        SOURCE KEY (from_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        DESTINATION KEY (to_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        LABEL {RelationRelatedTo},
                    {dvs.DVS_EDGES_IS_FROM_TABLE_NAME}
                        SOURCE KEY (from_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        DESTINATION KEY (to_node_id) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                        LABEL {RelationIsFrom}
                )
                """  # noqa: E501
            )

            conn.execute(create_table_sql)

        debug_print(
            create_table_sql,
            title=f"Creating table: '{dvs.DVS_GRAPH_TABLE_NAME}' with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )
        logger.info(f"✅ Created property graph: '{dvs.DVS_GRAPH_TABLE_NAME}'")
        return True

    @functools.cached_property
    def nodes(self) -> "Nodes":
        from dvs.db.graph.nodes.api import Nodes

        return Nodes(self.dvs)

    @functools.cached_property
    def edges(self) -> "Edges":
        from dvs.db.graph.edges.api import Edges

        return Edges(self.dvs)

    def get_neighbors(
        self,
        node_id_or_label: str | None = None,
        *,
        relation: RelationType | None = None,
        limit: int = 5,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[NodeType, EdgeType, NodeType]]:
        FROM_NODE_ALIAS = "from_node"
        TO_NODE_ALIAS = "to_node"
        RELATION_ALIAS = "rel"

        output: typing.List[typing.Tuple[NodeType, EdgeType, NodeType]] = []
        query_relations = (
            [RelationIsA, RelationHasA, RelationRelatedTo, RelationIsFrom]
            if relation is None
            else [relation]
        )
        conn = self.dvs.conn
        conn.execute(SQL_STMT_LOAD_DUCKPGQ)

        def run_query(
            query: str,
        ) -> typing.List[typing.Tuple[NodeType, EdgeType, NodeType]]:
            local_conn = conn.cursor()
            result = local_conn.execute(query)
            result_data = result.df().to_dict(orient="records")
            return [
                (
                    NodeType.model_validate(row[FROM_NODE_ALIAS]),
                    EdgeType.model_validate(row[RELATION_ALIAS]),
                    NodeType.model_validate(row[TO_NODE_ALIAS]),
                )
                for row in result_data
            ]

        with Timer() as timer:
            condition = (
                f" WHERE {FROM_NODE_ALIAS}.node_id = '{node_id_or_label}'"
                if node_id_or_label
                else ""
            )
            queries = [
                textwrap.dedent(
                    f"""
                    FROM GRAPH_TABLE (
                        {dvs.DVS_GRAPH_TABLE_NAME}
                        MATCH ({FROM_NODE_ALIAS}:nodes{condition})-[{RELATION_ALIAS}:{query_relation}]->({TO_NODE_ALIAS}:nodes)
                        COLUMNS ({FROM_NODE_ALIAS}, {RELATION_ALIAS}, {TO_NODE_ALIAS})
                    )
                    ORDER BY {FROM_NODE_ALIAS}.node_id
                    LIMIT {limit};
                    """  # noqa: E501
                )
                for query_relation in query_relations
            ]

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = executor.map(run_query, queries)
                for result in results:
                    output.extend(result)

        debug_print(
            "\n\n---\n\n".join(queries),
            title="Getting neighbors with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )
        return output
