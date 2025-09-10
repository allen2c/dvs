import logging
import textwrap
import typing
from concurrent.futures import ThreadPoolExecutor

import duckdb

import dvs
from dvs.types.edge import (
    RelationHasA,
    RelationIsA,
    RelationIsFrom,
    RelationRelatedTo,
    RelationType,
)
from dvs.types.node import Node as NodeType
from dvs.utils.debug_print import debug_print
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Algorithm:
    def __init__(self, dvs: dvs.DVS):
        """Graph algorithm API"""
        self.dvs = dvs

    def get_shortest_paths(
        self,
        from_node_id_or_label: str | None = None,
        to_node_id_or_label: str | None = None,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
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
        conn = conn or self.dvs.new_connection()

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
        conn: duckdb.DuckDBPyConnection | None = None,
        relation: RelationType,
        limit: int = 10,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[typing.Text, float]]:
        """Calculate local clustering coefficient for nodes in the graph."""
        output: typing.List[typing.Tuple[typing.Text, float]] = []

        conn = conn or self.dvs.new_connection()

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
        conn: duckdb.DuckDBPyConnection | None = None,
        relation: RelationType,
        limit: int = 15,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[typing.Text, int]]:
        """Find weakly connected components in the graph."""
        output: typing.List[typing.Tuple[typing.Text, int]] = []

        conn = conn or self.dvs.new_connection()

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
        conn: duckdb.DuckDBPyConnection | None = None,
        relation: RelationType,
        limit: int = 10,
        verbose: bool | None = None,
    ) -> typing.List[typing.Tuple[typing.Text, float]]:
        """Calculate PageRank scores for nodes in the graph."""

        output: typing.List[typing.Tuple[typing.Text, float]] = []

        conn = conn or self.dvs.new_connection()

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
