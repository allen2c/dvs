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
        """Initialize the property graph with required tables and extensions."""
        self.nodes.touch(verbose=self.dvs.v(verbose))
        self.edges.touch(verbose=self.dvs.v(verbose))

        with Timer() as timer:
            conn = self.dvs.new_connection()
            # First, execute the extension installation
            conn.cursor().execute(SQL_STMT_LOAD_DUCKPGQ)

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
            verbose=self.dvs.v(verbose),
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
        from_node_id_or_label: str | None = None,
        to_node_id_or_label: str | None = None,
        *,
        relation: RelationType | None = None,
        limit: int = 5,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[NodeType, EdgeType, NodeType]]:
        """Find neighboring nodes connected by edges with relation filtering."""
        FROM_NODE_ALIAS = "from_node"
        TO_NODE_ALIAS = "to_node"
        RELATION_ALIAS = "rel"

        output: typing.List[typing.Tuple[NodeType, EdgeType, NodeType]] = []
        query_relations = (
            [RelationIsA, RelationHasA, RelationRelatedTo, RelationIsFrom]
            if relation is None
            else [relation]
        )
        conn = self.dvs.new_connection()
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
            from_condition = (
                f" WHERE {FROM_NODE_ALIAS}.node_id = '{from_node_id_or_label}'"
                if from_node_id_or_label
                else ""
            )
            to_condition = (
                f" WHERE {TO_NODE_ALIAS}.node_id = '{to_node_id_or_label}'"
                if to_node_id_or_label
                else ""
            )
            queries = [
                textwrap.dedent(
                    f"""
                    FROM GRAPH_TABLE (
                        {dvs.DVS_GRAPH_TABLE_NAME}
                        MATCH ({FROM_NODE_ALIAS}:nodes{from_condition})-[{RELATION_ALIAS}:{query_relation}]->({TO_NODE_ALIAS}:nodes{to_condition})
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
            verbose=self.dvs.v(verbose),
        )
        return output

    def get_shortest_paths(
        self,
        from_node_id_or_label: str | None = None,
        to_node_id_or_label: str | None = None,
        *,
        relation: RelationType | None = None,
        limit: int = 15,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[NodeType, NodeType, int]]:
        """Find shortest paths between nodes using graph traversal algorithms."""
        FROM_NODE_ALIAS = "from_node"
        TO_NODE_ALIAS = "to_node"
        RELATION_ALIAS = "rel"
        DISTANCE_ALIAS = "distance"

        output: typing.List[typing.Tuple[NodeType, NodeType, int]] = []
        query_relations = (
            [RelationIsA, RelationHasA, RelationRelatedTo, RelationIsFrom]
            if relation is None
            else [relation]
        )
        conn = self.dvs.new_connection()
        conn.execute(SQL_STMT_LOAD_DUCKPGQ)

        def run_query(
            query: str,
        ) -> typing.List[typing.Tuple[NodeType, NodeType, int]]:
            local_conn = conn.cursor()
            result = local_conn.execute(query)
            result_data = result.df().to_dict(orient="records")
            return [
                (
                    NodeType.model_validate(row[FROM_NODE_ALIAS]),
                    NodeType.model_validate(row[TO_NODE_ALIAS]),
                    row[DISTANCE_ALIAS],
                )
                for row in result_data
            ]

        with Timer() as timer:
            from_condition = (
                f" WHERE {FROM_NODE_ALIAS}.node_id = '{from_node_id_or_label}'"
                if from_node_id_or_label
                else ""
            )
            to_condition = (
                f" WHERE {TO_NODE_ALIAS}.node_id = '{to_node_id_or_label}'"
                if to_node_id_or_label
                else ""
            )
            queries = [
                textwrap.dedent(
                    f"""
                    FROM GRAPH_TABLE (
                        {dvs.DVS_GRAPH_TABLE_NAME}
                        MATCH p = ANY SHORTEST ({FROM_NODE_ALIAS}:nodes{from_condition})-[{RELATION_ALIAS}:{query_relation}]->+({TO_NODE_ALIAS}:nodes{to_condition})
                        COLUMNS ({FROM_NODE_ALIAS}, {TO_NODE_ALIAS}, path_length(p) as {DISTANCE_ALIAS})
                    )
                    ORDER BY {DISTANCE_ALIAS}
                    LIMIT {limit};
                    """  # noqa: E501
                )
                for query_relation in query_relations
            ]

            with ThreadPoolExecutor(max_workers=4) as executor:
                results = executor.map(run_query, queries)
                for result in results:
                    output.extend(result)

        output.sort(key=lambda x: x[-1])
        output = output[:limit]

        debug_print(
            "\n\n---\n\n".join(queries),
            title="Getting shortest paths with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=self.dvs.v(verbose),
        )
        return output

    def local_clustering_coefficient(
        self,
        *,
        relation: RelationType,
        limit: int = 10,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[typing.Text, float]]:
        """Calculate local clustering coefficient for nodes in the graph."""
        output: typing.List[typing.Tuple[typing.Text, float]] = []

        conn = self.dvs.new_connection()
        conn.execute(SQL_STMT_LOAD_DUCKPGQ)

        with Timer() as timer:
            query = textwrap.dedent(
                f"""
                FROM local_clustering_coefficient(
                    {dvs.DVS_GRAPH_TABLE_NAME},
                    {dvs.DVS_NODES_TABLE_NAME},
                    {relation}
                )
                ORDER BY local_clustering_coefficient DESC
                LIMIT {limit}
                """  # noqa: E501
            ).strip()

            local_conn = conn.cursor()
            result = local_conn.execute(query)
            result_data = result.df().to_dict(orient="records")

            output = [
                (
                    row["node_id"],
                    float(row["local_clustering_coefficient"]),
                )
                for row in result_data
            ]

        debug_print(
            query,
            title="Local Clustering Coefficient Analysis:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=self.dvs.v(verbose),
        )
        return output

    def weakly_connected_component(
        self,
        *,
        relation: RelationType,
        limit: int = 15,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[typing.Text, int]]:
        """Find weakly connected components in the graph."""
        output: typing.List[typing.Tuple[typing.Text, int]] = []

        conn = self.dvs.new_connection()
        conn.execute(SQL_STMT_LOAD_DUCKPGQ)

        with Timer() as timer:
            query = textwrap.dedent(
                f"""
                FROM weakly_connected_component(
                    {dvs.DVS_GRAPH_TABLE_NAME},
                    {dvs.DVS_NODES_TABLE_NAME},
                    {relation}
                )
                ORDER BY componentId, node_id
                LIMIT {limit}
                """  # noqa: E501
            ).strip()

            local_conn = conn.cursor()
            result = local_conn.execute(query)
            result_data = result.df().to_dict(orient="records")

            output = [
                (
                    row["node_id"],
                    int(row["componentId"]),
                )
                for row in result_data
            ]

        debug_print(
            query,
            title="Weakly Connected Components Analysis:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=self.dvs.v(verbose),
        )
        return output

    def pagerank(
        self,
        *,
        relation: RelationType,
        limit: int = 10,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[typing.Text, float]]:
        """Calculate PageRank scores for nodes in the graph."""

        output: typing.List[typing.Tuple[typing.Text, float]] = []

        conn = self.dvs.new_connection()
        conn.execute(SQL_STMT_LOAD_DUCKPGQ)

        with Timer() as timer:
            query = textwrap.dedent(
                f"""
                FROM pagerank(
                    {dvs.DVS_GRAPH_TABLE_NAME},
                    {dvs.DVS_NODES_TABLE_NAME},
                    {relation}
                )
                ORDER BY pagerank DESC
                LIMIT {limit}
                """  # noqa: E501
            ).strip()

            local_conn = conn.cursor()
            result = local_conn.execute(query)
            result_data = result.df().to_dict(orient="records")

            output = [
                (
                    row["node_id"],
                    float(row["pagerank"]),
                )
                for row in result_data
            ]

        debug_print(
            query,
            title="PageRank Analysis:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=self.dvs.v(verbose),
        )
        return output
