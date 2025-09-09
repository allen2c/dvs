import asyncio
import functools
import logging
import textwrap
import typing
from concurrent.futures import ThreadPoolExecutor

from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel

import dvs
import dvs.utils.vss as VSS
from dvs.types.document import Document
from dvs.types.edge import Edge as EdgeType
from dvs.types.edge import (
    RelationHasA,
    RelationIsA,
    RelationIsFrom,
    RelationRelatedTo,
    RelationType,
)
from dvs.types.entity import Entity
from dvs.types.graphrag_result import GraphRAGResult
from dvs.types.node import Node as NodeType
from dvs.types.point import Point
from dvs.types.triplet import Triplet
from dvs.utils.debug_print import debug_print
from dvs.utils.sql_stmts import SQL_STMT_LOAD_DUCKPGQ
from dvs.utils.timer import Timer

if typing.TYPE_CHECKING:
    import networkx as nx

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

    def drop(self, *, verbose: bool | None = None) -> bool:
        self.nodes.drop(verbose=self.dvs.v(verbose))
        self.edges.drop(verbose=self.dvs.v(verbose))
        with Timer() as timer:
            conn = self.dvs.new_connection()
            conn.execute(SQL_STMT_LOAD_DUCKPGQ)
            conn.cursor().sql(
                f"DROP PROPERTY GRAPH IF EXISTS {dvs.DVS_GRAPH_TABLE_NAME}"
            )
        debug_print(
            f"DROP PROPERTY GRAPH IF EXISTS {dvs.DVS_GRAPH_TABLE_NAME}",
            title=f"Dropping property graph: '{dvs.DVS_GRAPH_TABLE_NAME}' with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=self.dvs.v(verbose),
        )
        return True

    async def rebuild_graph(
        self,
        *,
        chat_model: OpenAIResponsesModel | OpenAIChatCompletionsModel,
        document_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
        model_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
        verbose: bool | None = None,
    ) -> "nx.DiGraph":
        from dvs.utils.build_graph_from_documents import (
            canonical_map_from_triplets_entities,
            graph_from_triplets_entities,
            triplets_entities_from_document,
        )

        self.drop(verbose=self.dvs.v(verbose))
        self.touch(verbose=self.dvs.v(verbose))

        documents = [
            doc for doc in self.dvs.db.documents.gen(verbose=self.dvs.v(verbose))
        ]
        documents_task = [
            triplets_entities_from_document(
                document=document,
                chat_model=chat_model,
                extract_extra_entities=True,
                document_semaphore=document_semaphore,
                model_semaphore=model_semaphore,
                verbose=self.dvs.v(verbose),
            )
            for document in documents
        ]
        triplets_entities_results = await asyncio.gather(*documents_task)

        triplets: list[Triplet] = []
        entities: list[Entity] = []
        for result in triplets_entities_results:
            triplets.extend(result[0])
            entities.extend(result[1])

        canonical_map = await canonical_map_from_triplets_entities(
            triplets=triplets,
            entities=entities,
            embeddings_model=self.dvs.model,
            embeddings_model_settings=self.dvs.model_settings,
            chat_model=chat_model,
            model_semaphore=model_semaphore,
            verbose=self.dvs.v(verbose),
        )

        # Build the Final Knowledge Graph
        G, nodes, edges = await graph_from_triplets_entities(
            triplets=triplets,
            entities=entities,
            canonical_map=canonical_map,
        )
        self.nodes.bulk_create(nodes)
        self.edges.bulk_create(edges)

        logger.info("✨ Knowledge Graph Construction Complete! ✨")
        logger.info(f"Total Nodes: {len(nodes)}, Edges: {len(edges)}")
        return G

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

    # --- Internal helpers for graph expansions and scoring reuse ---
    def expand_is_a_bfs(
        self,
        seed_entities: set[str],
        max_hops: int,
        per_hop_limit: int,
        *,
        cap_total: int | None = None,
        verbose: bool | None = None,
    ) -> set[str]:
        """Expand entities via is_a in both directions using BFS."""
        if max_hops <= 0 or not seed_entities:
            return set(seed_entities)

        expanded: set[str] = set(seed_entities)
        frontier: set[str] = set(seed_entities)
        visited: set[str] = set(seed_entities)

        for _ in range(max_hops):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for entity_id in list(frontier):
                try:
                    outs = self.get_neighbors(
                        from_node_id_or_label=entity_id,
                        relation=RelationIsA,
                        limit=per_hop_limit,
                        verbose=self.dvs.v(verbose),
                    )
                except Exception:
                    outs = []
                try:
                    ins = self.get_neighbors(
                        to_node_id_or_label=entity_id,
                        relation=RelationIsA,
                        limit=per_hop_limit,
                        verbose=self.dvs.v(verbose),
                    )
                except Exception:
                    ins = []

                for from_node, _edge, to_node in list(outs) + list(ins):
                    other_node_id: str = (
                        to_node.node_id
                        if from_node.node_id == entity_id
                        else from_node.node_id
                    )
                    other_is_entity: bool = (
                        getattr(
                            to_node if from_node.node_id == entity_id else from_node,
                            "kind",
                            None,
                        )
                        == "entity"
                    )
                    if other_is_entity and other_node_id not in visited:
                        visited.add(other_node_id)
                        expanded.add(other_node_id)
                        next_frontier.add(other_node_id)
                        if cap_total is not None and len(expanded) >= cap_total:
                            return expanded
            frontier = next_frontier

        return expanded

    def expand_has_a_one_hop(
        self,
        entity_ids: set[str],
        per_entity_limit: int,
        *,
        cap_total: int | None = None,
        verbose: bool | None = None,
    ) -> set[str]:
        """Expand entities via has_a one hop in both directions."""
        if not entity_ids:
            return set()

        expanded: set[str] = set()
        seeds: list[str] = list(entity_ids)
        for eid in seeds:
            try:
                outs = self.get_neighbors(
                    from_node_id_or_label=eid,
                    relation=RelationHasA,
                    limit=per_entity_limit,
                    verbose=self.dvs.v(verbose),
                )
            except Exception:
                outs = []
            try:
                ins = self.get_neighbors(
                    to_node_id_or_label=eid,
                    relation=RelationHasA,
                    limit=per_entity_limit,
                    verbose=self.dvs.v(verbose),
                )
            except Exception:
                ins = []

            for from_node, _edge, to_node in list(outs) + list(ins):
                other = to_node if from_node.node_id == eid else from_node
                if getattr(other, "kind", None) == "entity":
                    expanded.add(other.node_id)
                    if cap_total is not None and len(expanded) >= cap_total:
                        return expanded

        return expanded

    def suppress_hubs_by_pagerank(
        self,
        entity_ids: set[str],
        top_percent: float,
        *,
        relation: RelationType = RelationRelatedTo,
        limit: int = 5000,
    ) -> set[str]:
        """Remove high-PageRank hubs above the given percentile cutoff."""
        if not entity_ids:
            return set()
        try:
            pr = self.pagerank(relation=relation, limit=limit, verbose=False)
            pr_map: dict[str, float] = {nid: sc for nid, sc in pr}
            if not pr_map:
                return set(entity_ids)
            scores: list[float] = sorted(pr_map.values(), reverse=True)
            pct: float = max(0.0, min(1.0, top_percent))
            idx: int = max(0, min(len(scores) - 1, int(len(scores) * pct) - 1))
            cutoff: float = scores[idx]
            return {eid for eid in entity_ids if pr_map.get(eid, 0.0) < cutoff}
        except Exception:
            return set(entity_ids)

    def collect_docs_via_is_from(
        self,
        entity_ids: set[str],
        per_entity_limit: int,
        *,
        cap_entities: int | None = None,
        verbose: bool | None = None,
    ) -> set[str]:
        """Collect document ids reachable via is_from from entities."""
        collected: set[str] = set()
        if not entity_ids:
            return collected

        processed: int = 0
        for eid in list(entity_ids):
            if cap_entities is not None and processed >= cap_entities:
                break
            processed += 1
            try:
                neighbors = self.get_neighbors(
                    from_node_id_or_label=eid,
                    relation=RelationIsFrom,
                    limit=per_entity_limit,
                    verbose=self.dvs.v(verbose),
                )
            except Exception:
                neighbors = []
            for from_node, _edge, to_node in neighbors:
                if getattr(to_node, "kind", None) == "document":
                    collected.add(to_node.label)
                if getattr(from_node, "kind", None) == "document":
                    collected.add(from_node.label)
        return collected

    def centroid_for_document(
        self, doc_id: str, *, limit_points: int = 5
    ) -> list[float]:
        """Compute centroid from a document's point embeddings."""
        vectors: list[list[float]] = []
        try:
            pts = self.dvs.db.points.gen(
                document_id=doc_id,
                limit=limit_points,
                with_embedding=True,
                verbose=False,
            )
            for p in pts:
                if p.embedding:
                    vectors.append(p.to_python())
        except Exception:
            pass
        from dvs.utils.graph_ops import mean_vector

        return mean_vector(vectors)

    async def embed_query_vector(self, query: str) -> list[float]:
        """Embed a query string and return a single vector."""
        return (
            await asyncio.to_thread(
                self.dvs.model.get_embeddings,
                query,
                model_settings=self.dvs.model_settings,
            )
        ).to_python()[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts and return vectors."""
        return (
            await asyncio.to_thread(
                self.dvs.model.get_embeddings,
                texts,
                model_settings=self.dvs.model_settings,
            )
        ).to_python()

    def collect_entities_via_is_from_for_documents(
        self,
        document_ids: list[str],
        *,
        limit_per_doc: int = 10,
        verbose: bool | None = None,
    ) -> set[str]:
        """Collect entities connected to documents via is_from edges."""
        entity_ids: set[str] = set()
        for doc_id in document_ids:
            try:
                doc_node = self.nodes.retrieve_by_label(
                    doc_id, verbose=self.dvs.v(verbose)
                )
                neighbors = self.get_neighbors(
                    to_node_id_or_label=doc_node.node_id,
                    relation=RelationIsFrom,
                    limit=limit_per_doc,
                    verbose=self.dvs.v(verbose),
                )
                for from_node, _edge, to_node in neighbors:
                    if getattr(from_node, "kind", None) == "entity":
                        entity_ids.add(from_node.node_id)
                    if getattr(to_node, "kind", None) == "entity":
                        entity_ids.add(to_node.node_id)
            except Exception:
                continue
        return entity_ids

    def gather_points_for_documents(
        self,
        document_ids: list[str] | set[str],
        *,
        per_doc_limit: int = 10,
        with_embedding: bool = True,
        verbose: bool | None = None,
    ) -> list[Point]:
        """Gather points for documents; optionally include embeddings."""
        points: list[Point] = []
        for doc_id in list(document_ids):
            try:
                pts = self.dvs.db.points.gen(
                    document_id=doc_id,
                    limit=per_doc_limit,
                    with_embedding=with_embedding,
                    verbose=self.dvs.v(verbose),
                )
                points.extend(pts)
            except Exception:
                continue
        return points

    def normalize_by_top(self, values: list[float]) -> list[float]:
        """Normalize values by top value; 0 if top is 0 or empty."""
        if not values:
            return []
        top: float = float(values[0]) if values else 0.0
        top = top if top > 0 else (max(values) if values else 0.0)
        if top <= 0:
            return [0.0 for _ in values]
        return [float(v) / float(top) for v in values]

    def expand_entities(
        self,
        seed_entities: set[str],
        *,
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        suppress_hubs: bool = True,
        hub_pagerank_top_percent: float = 0.1,
        cap_total: int | None = None,
        verbose: bool | None = None,
    ) -> set[str]:
        """Expand entities via is_a and has_a, then suppress hubs by PageRank."""
        expanded: set[str] = set(seed_entities)
        if is_a_max_hops > 0 and expanded:
            expanded = self.expand_is_a_bfs(
                expanded,
                is_a_max_hops,
                is_a_limit_per_hop,
                cap_total=cap_total,
                verbose=self.dvs.v(verbose),
            )
        if has_a_enabled and expanded:
            expanded |= self.expand_has_a_one_hop(
                expanded,
                has_a_limit_per_entity,
                cap_total=cap_total,
                verbose=self.dvs.v(verbose),
            )
        if suppress_hubs and expanded:
            expanded = self.suppress_hubs_by_pagerank(
                expanded,
                hub_pagerank_top_percent,
                relation=RelationRelatedTo,
                limit=5000,
            )
        return expanded

    def wrap_base_results_as_graphrag(
        self,
        base: list[tuple[Point, Document, float]],
        *,
        top_k: int,
        strategy: typing.Literal[
            "vector_expansion",
            "graph_guided",
            "hybrid_scoring",
            "iterative_refinement",
        ],
        graph_score_value: float | None = 0.0,
    ) -> list[GraphRAGResult]:
        """Convert vector-only results into GraphRAGResult with normalization."""
        max_score: float = float(base[0][2]) if base else 1.0
        out: list[GraphRAGResult] = []
        for i, (_p, doc, sc) in enumerate(base[:top_k]):
            val: float = float(sc)
            norm: float = (val / max_score) if max_score > 0 else 0.0
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=val,
                    vector_score=val if graph_score_value is not None else None,
                    graph_score=graph_score_value,
                    iterations=None,
                    rank=i + 1,
                    normalized_score=norm,
                    strategy=strategy,
                )
            )
        return out

    def get_graph_importance_for_document(
        self,
        document_id: str,
        *,
        pagerank_map: dict[str, float],
        max_pagerank: float,
        is_a_max_hops: int,
        is_a_limit_per_hop: int,
        has_a_enabled: bool,
        has_a_limit_per_entity: int,
        suppress_hubs: bool,
        hub_pagerank_top_percent: float,
    ) -> float:
        """Return normalized graph importance [0,1] for a document node."""
        if document_id in pagerank_map and max_pagerank > 0:
            return pagerank_map[document_id] / max_pagerank

        try:
            neighbors = self.get_neighbors(
                to_node_id_or_label=document_id,
                relation=RelationIsFrom,
                limit=30,
                verbose=False,
            )
            entities: set[str] = set()
            for _, _, to_node in neighbors:
                if getattr(to_node, "kind", None) == "entity":
                    entities.add(to_node.node_id)

            if is_a_max_hops > 0 and entities:
                entities = self.expand_is_a_bfs(
                    entities,
                    is_a_max_hops,
                    is_a_limit_per_hop,
                    verbose=False,
                )

            if has_a_enabled and entities:
                entities |= self.expand_has_a_one_hop(
                    entities,
                    has_a_limit_per_entity,
                    verbose=False,
                )

            if suppress_hubs and entities:
                entities = self.suppress_hubs_by_pagerank(
                    entities,
                    hub_pagerank_top_percent,
                    relation=RelationRelatedTo,
                    limit=5000,
                )

            best: float = 0.0
            for eid in entities:
                node_score = pagerank_map.get(eid, 0.0)
                if max_pagerank > 0:
                    best = max(best, node_score / max_pagerank)
            return best
        except Exception:
            return 0.0

    def get_graph_distance_score(
        self, document_id: str, original_doc_ids: list[str]
    ) -> float:
        """Distance score [0,1] derived from shortest path to seeds."""
        return self.calculate_graph_relevance(
            document_id,
            original_doc_ids,
            related_entities=set(),
            verbose=False,
        )

    async def search_vector_expansion(
        self,
        query: str,
        top_k: int = 3,
        *,
        graph_expansion_depth: int = 1,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        with_embedding: bool = False,
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        related_to_enabled: bool = True,
        related_to_limit_per_entity: int = 1,
        docs_per_entity_limit: int = 3,
        entity_expansion_cap: int = 200,
        suppress_hubs: bool = True,
        hub_pagerank_top_percent: float = 0.1,
        related_to_require_seed_touch: bool = True,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 1: vector search → graph expand → re-score (vector+graph).
        Expand seed entities via is_from, is_a, has_a (and optional related_to),
        collect more docs, then combine vector and graph relevance for ranking.

        Pros: simple; leverages entities near seed docs; explainable via edges.
        Cons: sensitive to initial recall; may drift and pull hubs; extra passes.
        Use when: you want quick recall boost from doc-seeded entities and moderate
        latency is acceptable; centrality metrics are unavailable.

        Diagram (Mermaid):
        ```mermaid
        flowchart TD
            Q[Query] --> VS[Vector Search x2 top-k]
            VS --> Seeds[Seed Documents]
            Seeds --> E0[Entities via is_from]
            E0 -->|is_a BFS| E1[Expanded Entities]
            E1 -->|has_a 1-hop| E2[Expanded Entities]
            E2 -->|related_to 1-hop optional| E3[Expanded Entities]
            E3 --> Docs[Collect Docs via is_from]
            Docs --> Cand[Candidate Points]
            Q --> Embed[Embed Query]
            Cand --> VSim[Vector Similarity]
            Docs --> GRel[Graph Relevance to Seeds]
            VSim --> Combine[Weighted Sum]
            GRel --> Combine
            Combine --> TopK[Top-k Results]
        ```
        """

        from dvs.utils.cosine_similarity import cosine_similarity

        # Step 1: Vector search - Find most relevant documents
        initial_results = await self.dvs.search(
            query=query,
            top_k=top_k * 2,  # Expand candidate set
            with_embedding=with_embedding,
            verbose=self.dvs.v(verbose),
        )

        if not initial_results:
            return []

        # Step 2: Find corresponding document nodes from document IDs
        document_ids = [doc.document_id for _, doc, _ in initial_results]
        document_nodes = []
        for doc_id in document_ids:
            try:
                doc_node = self.nodes.retrieve_by_label(
                    doc_id,
                    verbose=self.dvs.v(verbose),
                )
                document_nodes.append(doc_node)
            except Exception:
                continue

        # Step 3: Find related entity nodes through "is_from" relationship
        related_entities: set[str] = set()
        if document_ids:
            related_entities = self.collect_entities_via_is_from_for_documents(
                document_ids, limit_per_doc=10, verbose=self.dvs.v(verbose)
            )

        # Step 3b: Expand entities using is_a (multi-hop as synonym),
        #          has_a (1-hop), related_to (1-hop)
        #  - is_a: treat as equivalence; explore both directions up to is_a_max_hops
        #  - has_a: 1-hop in both directions; limited fanout
        #  - related_to: 1-hop in both directions; limited fanout (stricter)
        expanded_entity_ids: set[str] = set(related_entities)
        if is_a_max_hops > 0 and len(expanded_entity_ids) < entity_expansion_cap:
            expanded_entity_ids = self.expand_is_a_bfs(
                expanded_entity_ids,
                is_a_max_hops,
                is_a_limit_per_hop,
                cap_total=entity_expansion_cap,
                verbose=False,
            )
        if has_a_enabled and len(expanded_entity_ids) < entity_expansion_cap:
            expanded_entity_ids |= self.expand_has_a_one_hop(
                expanded_entity_ids,
                has_a_limit_per_entity,
                cap_total=entity_expansion_cap,
                verbose=False,
            )

        # related_to 1-hop (both directions, stricter fanout)
        if related_to_enabled and len(expanded_entity_ids) < entity_expansion_cap:
            seeds_for_related: list[str] = list(expanded_entity_ids)[
                :entity_expansion_cap
            ]
            seed_doc_ids_set: set[str] = set(document_ids)
            touch_cache: dict[str, bool] = {}

            def _entity_touches_seed(entity_id: str) -> bool:
                """Check if entity connects to any seed doc via is_from."""
                cached = touch_cache.get(entity_id)
                if cached is not None:
                    return cached
                ok: bool = False
                try:
                    neighbors = self.dvs.db.graph.get_neighbors(
                        from_node_id_or_label=entity_id,
                        relation=typing.cast(RelationType, RelationIsFrom),
                        limit=5,
                        verbose=False,
                    )
                    for _fn, _edge, to_node in neighbors:
                        is_doc: bool = getattr(to_node, "kind", None) == "document"
                        if is_doc and (to_node.label in seed_doc_ids_set):
                            ok = True
                            break
                except Exception:
                    ok = False
                touch_cache[entity_id] = ok
                return ok

            for eid in seeds_for_related:
                try:
                    out_neighbors = self.dvs.db.graph.get_neighbors(
                        from_node_id_or_label=eid,
                        relation=RelationRelatedTo,
                        limit=related_to_limit_per_entity,
                        verbose=False,
                    )
                except Exception:
                    out_neighbors = []
                try:
                    in_neighbors = self.dvs.db.graph.get_neighbors(
                        to_node_id_or_label=eid,
                        relation=RelationRelatedTo,
                        limit=related_to_limit_per_entity,
                        verbose=False,
                    )
                except Exception:
                    in_neighbors = []

                for from_node, _edge, to_node in list(out_neighbors) + list(
                    in_neighbors
                ):
                    other = to_node if from_node.node_id == eid else from_node
                    if getattr(other, "kind", None) == "entity":
                        needs_touch: bool = (
                            related_to_require_seed_touch
                            and not _entity_touches_seed(other.node_id)
                        )
                        if needs_touch:
                            continue
                        expanded_entity_ids.add(other.node_id)
                        if len(expanded_entity_ids) >= entity_expansion_cap:
                            break
                if len(expanded_entity_ids) >= entity_expansion_cap:
                    break

        # Counters for logging
        seeds_count: int = len(related_entities)
        pre_hub_entities_count: int = len(expanded_entity_ids)

        # Hub suppression using PageRank over related_to graph
        if suppress_hubs and expanded_entity_ids:
            try:
                expanded_entity_ids = self.suppress_hubs_by_pagerank(
                    expanded_entity_ids,
                    hub_pagerank_top_percent,
                    relation=RelationRelatedTo,
                    limit=5000,
                )
            except Exception:
                pass

        # Step 4: Graph expansion on these entity nodes to find more related documents
        expanded_document_ids: set[str] = set(document_ids)
        if graph_expansion_depth > 0:
            expanded_document_ids |= self.collect_docs_via_is_from(
                set(list(expanded_entity_ids)[: min(len(expanded_entity_ids), 200)]),
                per_entity_limit=docs_per_entity_limit,
                verbose=self.dvs.v(verbose),
            )

        # Compute summary counts and log (verbose)
        suppressed_count: int = pre_hub_entities_count - len(expanded_entity_ids)
        added_docs_count: int = len(expanded_document_ids.difference(set(document_ids)))
        if self.dvs.v(verbose):
            logger.info(
                (
                    f"[S1] seeds={seeds_count} "
                    f"entities_after={len(expanded_entity_ids)} "
                    f"suppressed={suppressed_count} "
                    f"docs_added={added_docs_count}"
                )
            )

        # Step 5: Re-vector search on expanded document collection
        expanded_candidates = []
        for point in self.gather_points_for_documents(
            expanded_document_ids,
            per_doc_limit=10,
            with_embedding=True,
            verbose=self.dvs.v(verbose),
        ):
            if point.embedding:
                expanded_candidates.append(point)

        # Fall back to original results if no candidates found
        if not expanded_candidates:
            # Fallback: wrap initial results into GraphRAGResult
            wrapped: list[GraphRAGResult] = []
            # Normalize by top score if possible
            max_score: float = float(initial_results[0][2]) if initial_results else 1.0
            for idx, (_pt, doc, sc) in enumerate(initial_results[:top_k], start=1):
                val: float = float(sc)
                norm: float = (val / max_score) if max_score > 0 else 0.0
                wrapped.append(
                    GraphRAGResult(
                        document=doc,
                        score=val,
                        vector_score=val,
                        graph_score=0.0,
                        iterations=None,
                        rank=idx,
                        normalized_score=norm,
                        strategy="vector_expansion",
                    )
                )
            return wrapped

        # Step 6: Calculate vector similarity for candidate points
        query_vector: list[float] = await self.embed_query_vector(query)

        scored_candidates = []
        for point in expanded_candidates:
            try:
                point_vector = point.to_python()
                # Calculate cosine similarity
                similarity = cosine_similarity(query_vector, point_vector)
                scored_candidates.append((point, similarity))
            except Exception:
                continue

        # Step 7: Calculate combined score (vector similarity + graph relevance)
        final_results: list[tuple[Point, Document, float, float, float]] = []
        for point, vector_score in scored_candidates:
            try:
                doc = self.dvs.db.documents.retrieve(
                    point.document_id, verbose=self.dvs.v(verbose)
                )

                # Calculate graph relevance: based on distance to original
                # query-related documents
                graph_score = self.calculate_graph_relevance(
                    point.document_id,
                    document_ids,
                    related_entities,
                    verbose=self.dvs.v(verbose),
                )

                # Combined scoring
                combined_score: float = vector_weight * float(
                    vector_score
                ) + graph_weight * float(graph_score)

                final_results.append(
                    (
                        point,
                        doc,
                        combined_score,
                        float(vector_score),
                        float(graph_score),
                    )
                )

            except Exception:
                continue

        # Step 8: Sort by combined score and return top-k
        final_results.sort(key=lambda x: x[2], reverse=True)
        top = final_results[:top_k]
        # Normalize scores by top score
        max_score: float = float(top[0][2]) if top else 1.0
        out: list[GraphRAGResult] = []
        for idx, (_p, doc, combined, vsc, gsc) in enumerate(top, start=1):
            norm: float = (float(combined) / max_score) if max_score > 0 else 0.0
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=float(combined),
                    vector_score=float(vsc),
                    graph_score=float(gsc),
                    iterations=None,
                    rank=idx,
                    normalized_score=norm,
                    strategy="vector_expansion",
                )
            )
        return out

    async def search_graph_guided(
        self,
        query: str,
        top_k: int = 3,
        *,
        relation_types: list[str] | None = None,
        centrality_threshold: float = 0.5,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        with_embedding: bool = False,
        # Expansion and hub controls
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        related_to_limit_per_entity: int = 1,
        suppress_hubs: bool = True,
        hub_pagerank_top_percent: float = 0.1,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 2: PageRank-guided vector search over salient entities.
        Select high-centrality entities, expand lightly, collect related docs,
        then combine vector similarity with graph importance for ranking.

        Pros: robust for ambiguous queries; targets globally important graph areas.
        Cons: depends on centrality quality; may miss niche items; algo overhead.
        Use when: graph is rich and centrality is meaningful; discovery-oriented
        retrieval benefits from salient-node guidance.

        Diagram (Mermaid):
        ```mermaid
        flowchart TD
            Q[Query] --> PR[PageRank RelatedTo]
            PR --> Important[Select Important Entities]
            Important -->|is_a BFS| E1[Expanded Entities]
            E1 -->|has_a 1-hop| E2[Expanded Entities]
            E2 --> Docs[Collect Docs via is_from]
            Docs --> Cand[Candidate Points]
            Q --> Embed[Embed Query]
            Cand --> VSim[Vector Similarity]
            Important --> GImp[Graph Importance]
            VSim --> Combine[Weighted Sum]
            GImp --> Combine
            Combine --> TopK[Top-k Results]
        ```
        """
        # no direct Relation import needed; using existing helpers
        from dvs.utils.cosine_similarity import cosine_similarity

        # Step 1: Find important nodes using PageRank
        logger.debug("📊 Step 1: Finding important nodes using PageRank...")

        # Use PageRank to identify central nodes in the graph
        relation_type = RelationRelatedTo
        if relation_types:
            # Map string to proper RelationType
            relation_map = {
                "is_a": "is_a",
                "has_a": "has_a",
                "related_to": "related_to",
                "is_from": "is_from",
            }
            if relation_types[0] in relation_map:
                relation_type = relation_types[0]  # type: ignore

        pagerank_results = self.dvs.db.graph.pagerank(
            relation=relation_type,  # type: ignore
            limit=top_k * 10,  # Get more candidates for filtering
            verbose=self.dvs.v(verbose),
        )

        if not pagerank_results:
            logger.warning(
                "⚠️ No PageRank results found, falling back to regular search"
            )
            base = await self.dvs.search(
                query,
                top_k=top_k,
                with_embedding=with_embedding,
                verbose=self.dvs.v(verbose),
            )
            max_score: float = float(base[0][2]) if base else 1.0
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=0.0,
                    iterations=None,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score) if max_score > 0 else 0.0,
                    strategy="graph_guided",
                )
                for i, (_p, doc, sc) in enumerate(base)
            ]

        # Filter to get high-centrality nodes above threshold
        important_nodes = []
        max_pagerank = (
            max(score for _, score in pagerank_results) if pagerank_results else 1.0
        )

        for node_id, pagerank_score in pagerank_results:
            # Normalize score and filter by centrality threshold
            normalized_score = pagerank_score / max_pagerank if max_pagerank > 0 else 0
            if normalized_score >= centrality_threshold:
                important_nodes.append((node_id, normalized_score))

        if not important_nodes:
            logger.warning(
                "⚠️ No nodes above centrality threshold, using top-ranked nodes"
            )
            # Fallback: use top nodes regardless of threshold
            important_nodes = [
                (node_id, score / max_pagerank)
                for node_id, score in pagerank_results[: top_k * 3]
            ]

        logger.info(
            f"✅ Found {len(important_nodes)} important nodes for vector search"
        )

        # Step 2: Get documents related to these important nodes
        logger.debug("🔍 Step 2: Performing vector search on important nodes...")
        # Expand entities: is_a multi-hop and has_a 1-hop
        expanded_entities: set[str] = {nid for (nid, _score) in important_nodes}

        # is_a BFS both directions
        if is_a_max_hops > 0:
            frontier: set[str] = set(expanded_entities)
            visited: set[str] = set(expanded_entities)
            for _ in range(is_a_max_hops):
                if not frontier:
                    break
                next_frontier: set[str] = set()
                for eid in list(frontier):
                    try:
                        outs = self.dvs.db.graph.get_neighbors(
                            from_node_id_or_label=eid,
                            relation=RelationIsA,
                            limit=is_a_limit_per_hop,
                            verbose=False,
                        )
                    except Exception:
                        outs = []
                    try:
                        ins = self.dvs.db.graph.get_neighbors(
                            to_node_id_or_label=eid,
                            relation=RelationIsA,
                            limit=is_a_limit_per_hop,
                            verbose=False,
                        )
                    except Exception:
                        ins = []
                    for fn, _e, tn in list(outs) + list(ins):
                        other = tn if fn.node_id == eid else fn
                        if getattr(other, "kind", None) == "entity":
                            oid: str = other.node_id
                            if oid not in visited:
                                visited.add(oid)
                                expanded_entities.add(oid)
                                next_frontier.add(oid)
                frontier = next_frontier

        # has_a 1-hop both directions
        if has_a_enabled and expanded_entities:
            seeds: list[str] = list(expanded_entities)[:500]
            for eid in seeds:
                try:
                    outs = self.dvs.db.graph.get_neighbors(
                        from_node_id_or_label=eid,
                        relation=RelationHasA,
                        limit=has_a_limit_per_entity,
                        verbose=False,
                    )
                except Exception:
                    outs = []
                try:
                    ins = self.dvs.db.graph.get_neighbors(
                        to_node_id_or_label=eid,
                        relation=RelationHasA,
                        limit=has_a_limit_per_entity,
                        verbose=False,
                    )
                except Exception:
                    ins = []
                for fn, _e, tn in list(outs) + list(ins):
                    other = tn if fn.node_id == eid else fn
                    if getattr(other, "kind", None) == "entity":
                        expanded_entities.add(other.node_id)

        # Hub suppression on expanded entities
        if suppress_hubs and expanded_entities:
            try:
                pr = self.dvs.db.graph.pagerank(
                    relation=RelationRelatedTo, limit=5000, verbose=False
                )
                pr_map: dict[str, float] = {nid: sc for nid, sc in pr}
                if pr_map:
                    scores: list[float] = sorted(pr_map.values(), reverse=True)
                    pct: float = max(0.0, min(1.0, hub_pagerank_top_percent))
                    idx: int = max(0, min(len(scores) - 1, int(len(scores) * pct) - 1))
                    cutoff: float = scores[idx]
                    expanded_entities = {
                        eid
                        for eid in expanded_entities
                        if pr_map.get(eid, 0.0) < cutoff
                    }
            except Exception:
                pass

        # Collect documents from expanded entities via is_from
        related_documents = set()
        for node_id in list(expanded_entities)[:1000]:
            try:
                neighbors = self.dvs.db.graph.get_neighbors(
                    from_node_id_or_label=node_id,
                    relation=RelationIsFrom,
                    limit=5,
                    verbose=self.dvs.v(verbose),
                )
                for _, _, doc_node in neighbors:
                    if doc_node.kind == "document":
                        related_documents.add(doc_node.label)
            except Exception as e:
                logger.error(f"⚠️ Error getting neighbors for node {node_id}: {e}")
                continue

        # Summary logging (standardized)
        seeds_count_s2: int = len({nid for (nid, _sc) in important_nodes})
        entities_count_s2: int = len(expanded_entities)
        docs_count_s2: int = len(related_documents)
        suppressed_count_s2: int = 0
        if self.dvs.v(verbose):
            logger.info(
                (
                    f"[S2] seeds={seeds_count_s2} "
                    f"entities_after={entities_count_s2} "
                    f"suppressed={suppressed_count_s2} "
                    f"docs_collected={docs_count_s2}"
                )
            )

        if not related_documents:
            logger.warning(
                "⚠️ No related documents found, falling back to regular search"
            )
            base = await self.dvs.search(
                query,
                top_k=top_k,
                with_embedding=with_embedding,
                verbose=self.dvs.v(verbose),
            )
            max_score: float = float(base[0][2]) if base else 1.0
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=0.0,
                    iterations=None,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score) if max_score > 0 else 0.0,
                    strategy="graph_guided",
                )
                for i, (_p, doc, sc) in enumerate(base)
            ]

        # Step 3: Vector search on the collected documents
        logger.debug(
            f"📈 Step 3: Vector search on {len(related_documents)} documents..."
        )

        # Get all points for the related documents
        candidate_points: list["Point"] = self.gather_points_for_documents(
            list(related_documents)[:50],
            per_doc_limit=10,
            with_embedding=True,
            verbose=self.dvs.v(verbose),
        )

        if not candidate_points:
            logger.warning(
                "⚠️ No candidate points found, falling back to regular search"
            )
            base = await self.dvs.search(
                query,
                top_k=top_k,
                with_embedding=with_embedding,
                verbose=self.dvs.v(verbose),
            )
            max_score: float = float(base[0][2]) if base else 1.0
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=0.0,
                    iterations=None,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score) if max_score > 0 else 0.0,
                    strategy="graph_guided",
                )
                for i, (_p, doc, sc) in enumerate(base)
            ]

        # Step 4: Calculate vector similarities and combine with graph scores
        logger.debug("⚖️ Step 4: Calculating combined scores...")

        # Get query embedding
        query_vector: list[float] = await self.embed_query_vector(query)

        scored_candidates: list[tuple[Point, Document, float, float]] = []
        for point in candidate_points:
            try:
                if not point.embedding:
                    raise ValueError("Point has no embedding")
                    continue

                point_vector = point.to_python()
                vector_similarity = cosine_similarity(query_vector, point_vector)

                # Find the document this point belongs to
                doc = self.dvs.db.documents.retrieve(point.document_id, verbose=False)

                # Get graph importance score for this document's related entities
                graph_importance = 0.0
                for node_id, node_score in important_nodes:
                    try:
                        # Check if this document is related to the important entity
                        # Match by document label, not node_id
                        doc_node = self.dvs.db.graph.nodes.retrieve_by_label(
                            doc.document_id, verbose=False
                        )
                        neighbors = self.dvs.db.graph.get_neighbors(
                            from_node_id_or_label=node_id,
                            to_node_id_or_label=doc_node.node_id,
                            relation=RelationIsFrom,
                            limit=1,
                            verbose=False,
                        )
                        if neighbors:
                            graph_importance = max(graph_importance, node_score)
                    except Exception:
                        continue

                # Combined score: weighted average of vector similarity
                # and graph importance
                combined_score = (
                    vector_weight * vector_similarity + graph_weight * graph_importance
                )

                scored_candidates.append(
                    (point, doc, combined_score, vector_similarity)
                )

            except Exception as e:
                logger.error(f"⚠️ Error processing point {point.point_id}: {e}")
                continue

        # Step 5: Return top-k results
        scored_candidates.sort(key=lambda x: x[2], reverse=True)

        logger.info(
            f"✅ Graph-Guided Vector Search completed. "
            f"Found {len(scored_candidates)} candidates."
        )

        top = scored_candidates[:top_k]
        max_score: float = float(top[0][2]) if top else 1.0
        out: list[GraphRAGResult] = []
        for idx, (_p, doc, combined, vsc) in enumerate(top, start=1):
            norm: float = (float(combined) / max_score) if max_score > 0 else 0.0
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=float(combined),
                    vector_score=float(vsc),
                    graph_score=None,
                    iterations=None,
                    rank=idx,
                    normalized_score=norm,
                    strategy="graph_guided",
                )
            )
        return out

    async def search_hybrid_scoring(
        self,
        query: str,
        top_k: int = 3,
        *,
        vector_weight: float = 0.5,
        graph_importance_weight: float = 0.3,
        graph_distance_weight: float = 0.2,
        with_embedding: bool = False,
        # Expansion and hub controls
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        suppress_hubs: bool = True,
        hub_pagerank_top_percent: float = 0.1,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 3: weighted blend of vector score, graph importance, distance.
        Start with vector candidates, add PageRank-based importance and
        shortest-path distance to seed docs, then combine by weights.

        Pros: balances precision/recall; uses structure+semantics; explainable parts.
        Cons: needs weight tuning; adds PR/shortest-path cost; double-count risk.
        Use when: you can calibrate weights offline and want stable cross-domain
        performance.

        Diagram (Mermaid):
        ```mermaid
        flowchart TD
            Q[Query] --> VS[Vector Search x3 top-k]
            VS --> Seeds[Original Docs]
            Seeds --> OIDs[Original Doc IDs]
            PR[PageRank RelatedTo] --> ImpMap[Importance Map]
            OIDs --> Dist[Shortest Path Distance]
            VS --> Cand[Candidates]
            Cand --> VScore[Vector Score]
            ImpMap --> GImp[Graph Importance]
            Dist --> GDist[Graph Distance]
            VScore --> Combine[Weighted Sum]
            GImp --> Combine
            GDist --> Combine
            Combine --> TopK[Top-k Results]
        ```
        """
        # using helpers; no direct Relation imports needed here

        # cosine_similarity not needed in Strategy 3
        # Step 1: Initial vector search to get candidate points
        initial_results: list[tuple["Point", "Document", float]] = (
            await self.dvs.search(
                query=query,
                top_k=max(1, top_k * 3),
                with_embedding=with_embedding,
                verbose=self.dvs.v(verbose),
            )
        )

        if not initial_results:
            return []

        original_doc_ids: list[str] = [doc.document_id for _, doc, _ in initial_results]

        # Step 2: Prepare PageRank-based graph importance
        pagerank_results = self.dvs.db.graph.pagerank(
            relation=RelationRelatedTo,  # Use general semantic connectivity
            limit=top_k * 50,
            verbose=self.dvs.v(verbose),
        )

        pagerank_map: dict[str, float] = {
            node_id: score for node_id, score in pagerank_results
        }
        max_pagerank: float = max(pagerank_map.values()) if pagerank_map else 1.0

        def get_graph_importance_for_document(document_id: str) -> float:
            return self.get_graph_importance_for_document(
                document_id,
                pagerank_map=pagerank_map,
                max_pagerank=max_pagerank,
                is_a_max_hops=is_a_max_hops,
                is_a_limit_per_hop=is_a_limit_per_hop,
                has_a_enabled=has_a_enabled,
                has_a_limit_per_entity=has_a_limit_per_entity,
                suppress_hubs=suppress_hubs,
                hub_pagerank_top_percent=hub_pagerank_top_percent,
            )

        # Step 3: Compute distance-based score using shortest paths to originals
        # Reuse _calculate_graph_relevance which maps distance to [0,1]
        def get_graph_distance_score(document_id: str) -> float:
            return self.get_graph_distance_score(document_id, original_doc_ids)

        # Step 4: Combine scores
        combined_results: list[tuple[Point, Document, float, float, float]] = []

        for point, doc, vector_score in initial_results:
            try:
                graph_importance: float = get_graph_importance_for_document(
                    doc.document_id
                )
                graph_distance_score: float = get_graph_distance_score(doc.document_id)

                combined_score: float = (
                    vector_weight * float(vector_score)
                    + graph_importance_weight * graph_importance
                    + graph_distance_weight * graph_distance_score
                )

                combined_results.append(
                    (
                        point,
                        doc,
                        combined_score,
                        float(graph_importance),
                        float(graph_distance_score),
                    )
                )
            except Exception:
                continue

        # Step 5: Sort and return top-k
        combined_results.sort(key=lambda x: x[2], reverse=True)
        top = combined_results[:top_k]
        max_score: float = float(top[0][2]) if top else 1.0
        out: list[GraphRAGResult] = []
        for idx, (_p, doc, combined, gimp, gdist) in enumerate(top, start=1):
            norm: float = (float(combined) / max_score) if max_score > 0 else 0.0
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=float(combined),
                    vector_score=None,
                    graph_score=float(0.5 * (gimp + gdist)),
                    iterations=None,
                    rank=idx,
                    normalized_score=norm,
                    strategy="hybrid_scoring",
                )
            )
        return out

    async def search_iterative_refinement(
        self,
        query: str,
        top_k: int = 3,
        *,
        max_iterations: int = 3,
        refinement_threshold: float = 0.05,
        expansions_per_iter: int = 3,
        query_expander: (
            typing.Callable[[str], "typing.Awaitable[list[str]]"] | None
        ) = None,
        # Expansion and hub controls
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        suppress_hubs: bool = True,
        hub_pagerank_top_percent: float = 0.1,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 4: iterate expand → embed → search until gains are small.
        LLM expands queries; graph adds new docs via entities; merge vectors and
        re-search; stop on low improvement or max iterations.

        Pros: escapes local minima; adapts via feedback; good for exploration.
        Cons: multi-round latency; expander quality critical; tuning convergence.
        Use when: batch/offline or higher-latency is fine; need high recall and
        adaptive retrieval.

        Diagram (Mermaid):
        ```mermaid
        flowchart TD
            Q[Query] --> Embed0[Embed Base]
            Embed0 --> VS0[Baseline Vector Search]
            VS0 --> Loop{Improvement > threshold and iters < max?}
            Loop -- Yes --> LLM[LLM Expand Queries]
            LLM --> EmbedX[Embed Expansions]
            VS0 --> Seeds[Best Docs]
            Seeds --> Ent[Entities via is_from]
            Ent -->|is_a/has_a| E[Expanded Entities]
            E --> Docs[Collect Docs]
            Docs --> GVecs[Graph Vectors]
            Embed0 --> Cmp[Combine Vectors]
            EmbedX --> Cmp
            GVecs --> Cmp
            Cmp --> VS1[Refined Vector Search]
            VS1 --> Loop
            Loop -- No --> Out[Best Results]
        ```
        """
        # Relation imports not needed; using Graph helpers for expansion/suppression

        if query_expander is None:
            raise ValueError("query_expander must be provided for LLM-based expansion.")

        # Step 0: Prepare baseline using original query
        base_vector: list[float] = await self.embed_query_vector(query)

        baseline_results = await VSS.vector_search(
            vector=base_vector,
            top_k=max(1, top_k),
            embedding_dimensions=self.dvs.db_manifest.embedding_dimensions,
            documents_table_name=dvs.DVS_DOCUMENTS_TABLE_NAME,
            points_table_name=dvs.DVS_POINTS_TABLE_NAME,
            conn=self.dvs.new_connection(read_only=True),
            with_embedding=False,
            debug=self.dvs.v(verbose),
            console=self.dvs.settings.console,
        )

        if not baseline_results:
            return []

        best_results: list[tuple[Point, Document, float]] = baseline_results
        best_top1: float = float(baseline_results[0][2])
        current_vector: list[float] = list(base_vector)
        iterations: int = 0

        if self.dvs.v(verbose):
            top_docs = ", ".join(
                [
                    doc.name
                    for _, doc, _ in baseline_results[: min(3, len(baseline_results))]
                ]
            )
            logger.info(
                f"[gRAG_Iter] Baseline top1={best_top1:.3f}; top docs: {top_docs}"
            )

        # Iterative loop
        while iterations < max_iterations:
            # 1) Expand query terms with LLM agent
            expanded_queries: list[str] = [
                q
                for q in (await query_expander(query))[:expansions_per_iter]
                if isinstance(q, str) and q.strip()
            ]
            if not expanded_queries:
                logger.info("[gRAG_Iter] No expanded queries returned; stopping.")
                break

            logger.debug(
                f"[gRAG_Iter] Iter {iterations + 1} expansions: {expanded_queries}"
            )

            # 2) Embed expansions
            expanded_vectors = await self.embed_texts(expanded_queries)

            # 2b) Graph-guided expansion (is_from 1-hop) to boost recall
            seed_doc_ids: list[str] = [doc.document_id for _, doc, _ in best_results]
            entity_ids: set[str] = self.collect_entities_via_is_from_for_documents(
                seed_doc_ids, limit_per_doc=300, verbose=False
            )

            # Expand entities via is_a multi-hop, then has_a 1-hop
            expanded_entities_iter: set[str] = self.expand_entities(
                set(entity_ids),
                is_a_max_hops=is_a_max_hops,
                is_a_limit_per_hop=is_a_limit_per_hop,
                has_a_enabled=has_a_enabled,
                has_a_limit_per_entity=has_a_limit_per_entity,
                suppress_hubs=suppress_hubs,
                hub_pagerank_top_percent=hub_pagerank_top_percent,
                cap_total=None,
                verbose=False,
            )

            # Expand to new documents (caps: 8 per entity, 150 total)
            graph_doc_ids: set[str] = self.collect_docs_via_is_from(
                expanded_entities_iter,
                per_entity_limit=8,
                cap_entities=150,
                verbose=False,
            )

            # Summary logging for S4
            if self.dvs.v(verbose):
                logger.info(
                    (
                        f"[S4] entities_seed={len(entity_ids)} "
                        f"entities_after={len(expanded_entities_iter)} "
                        f"docs_added={len(graph_doc_ids)}"
                    )
                )

            new_graph_docs = graph_doc_ids.difference(set(seed_doc_ids))
            if len(new_graph_docs) == 0:
                logger.info(
                    "[gRAG_Iter] No new documents from graph expansion; stopping."
                )
                break

            # Build centroid from candidate document points
            graph_vectors: list[list[float]] = []
            pts = self.gather_points_for_documents(
                list(new_graph_docs)[:150],
                per_doc_limit=3,
                with_embedding=True,
                verbose=False,
            )
            for pt in pts:
                if pt.embedding:
                    graph_vectors.append(pt.to_python())

            # 3) Combine vectors (weighted average: base, LLM, graph)
            has_llm = len(expanded_vectors) > 0
            has_graph = len(graph_vectors) > 0
            llm_w: float = 0.6 if has_llm else 0.0
            graph_w: float = 0.4 if has_graph else 0.0
            base_w: float = 1.0

            def mean_at(i: int, vecs: list[list[float]]) -> float:
                return sum(v[i] for v in vecs) / float(len(vecs)) if vecs else 0.0

            denom: float = base_w + llm_w + graph_w
            combined_vector: list[float] = []
            for i in range(len(current_vector)):
                val = (
                    base_w * current_vector[i]
                    + llm_w * mean_at(i, expanded_vectors)
                    + graph_w * mean_at(i, graph_vectors)
                ) / (denom if denom > 0 else 1.0)
                combined_vector.append(val)

            # 4) Re-search with refined vector
            refined_results = await VSS.vector_search(
                vector=combined_vector,
                top_k=max(1, top_k),
                embedding_dimensions=self.dvs.db_manifest.embedding_dimensions,
                documents_table_name=dvs.DVS_DOCUMENTS_TABLE_NAME,
                points_table_name=dvs.DVS_POINTS_TABLE_NAME,
                conn=self.dvs.new_connection(read_only=True),
                with_embedding=False,
                debug=self.dvs.v(verbose),
                console=self.dvs.settings.console,
            )

            if not refined_results:
                logger.info("[gRAG_Iter] Refined search returned no results; stopping.")
                break

            refined_top1: float = float(refined_results[0][2])
            improvement: float = refined_top1 - best_top1

            top_docs_refined = ", ".join(
                [
                    doc.name
                    for _, doc, _ in refined_results[: min(3, len(refined_results))]
                ]
            )
            logger.debug(
                (
                    f"[gRAG_Iter] Iter {iterations + 1} top1={refined_top1:.3f} "
                    + f"improve={improvement:.3f}; top docs: {top_docs_refined}"
                )
            )

            # 5) Check convergence using top-1 score improvement
            if improvement <= refinement_threshold:
                logger.info(
                    (
                        f"[gRAG_Iter] Stop: improvement {improvement:.3f} "
                        + f"<= threshold {refinement_threshold:.3f}."
                    )
                )
                break

            # 6) Accept refinement and continue
            best_results = refined_results
            best_top1 = refined_top1
            current_vector = combined_vector
            iterations += 1

            logger.info(
                f"[gRAG_Iter] Accept iter {iterations}; new best_top1={best_top1:.3f}"
            )

        # Build GraphRAGResult output
        output: list[GraphRAGResult] = [
            GraphRAGResult(
                document=doc,
                score=float(score),
                vector_score=float(score),
                graph_score=None,
                iterations=iterations,
            )
            for (_point, doc, score) in best_results[:top_k]
        ]

        return output

    async def search_context_aware(
        self,
        query: str,
        top_k: int = 3,
        *,
        context_similarity_threshold: float = 0.7,
        max_expansion_steps: int = 2,
        with_embedding: bool = False,
        # Expansion and hub controls
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        suppress_hubs: bool = True,
        hub_pagerank_top_percent: float = 0.1,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 5: baseline seeds → is_from expand → context-centroid ranking.
        Build a centroid from seed points, expand via entities, filter by context
        similarity, then rank with a blend of query and context similarity.

        Pros: low-latency; mitigates semantic drift via context filter; explainable.
        Cons: needs embeddings; sparse graphs reduce gains; threshold sensitive.
        Use when: you prefer contextual precision with tight latency and minimal
        global graph prerequisites; hub suppression optional.

        Diagram (Mermaid):
        ```mermaid
        flowchart TD
            Q[Query] --> VS[Baseline Vector Search]
            VS --> Seeds[Seed Points + Docs]
            Seeds --> Ctx[Compute Context Centroid]
            Seeds --> Ent[Entities via is_from]
            Ent -->|is_a/has_a| E[Expanded Entities]
            E --> Docs[Collect Candidate Docs]
            Docs --> Cent[Doc Centroids]
            Q --> Embed[Embed Query]
            Cent --> QSim[Query vs Doc Similarity]
            Ctx --> CSim[Context vs Doc Similarity]
            QSim --> Filter[Context Threshold]
            CSim --> Filter
            Filter --> Rank[Rank and Normalize]
            Rank --> TopK[Top-k Results]
        ```
        """
        from dvs.utils.cosine_similarity import cosine_similarity

        # 0) Baseline vector search to get seed context
        baseline_results: list[tuple[Point, Document, float]] = await self.dvs.search(
            query=query,
            top_k=max(1, top_k),
            with_embedding=True,  # need embeddings to compute centroids
            verbose=self.dvs.v(verbose),
        )

        if not baseline_results:
            return []

        # 1) Build context centroid from seed points

        seed_vectors: list[list[float]] = []
        seed_doc_ids: list[str] = []
        for pt, doc, _ in baseline_results:
            if pt.embedding:
                seed_vectors.append(pt.to_python())
            seed_doc_ids.append(doc.document_id)

        # If no embeddings returned, fetch a few points per doc with embeddings
        if not seed_vectors:
            for _pt, doc, _ in baseline_results:
                try:
                    pts = self.dvs.db.points.gen(
                        document_id=doc.document_id,
                        limit=3,
                        with_embedding=True,
                        verbose=False,
                    )
                    for p in pts:
                        if p.embedding:
                            seed_vectors.append(p.to_python())
                except Exception:
                    continue

        if self.dvs.v(verbose):
            logger.debug(
                (
                    f"[S5] seeds={len(baseline_results)} "
                    + f"seed_vecs(before_fetch)={len(seed_vectors)}"
                )
            )

        # If still empty, fallback return baseline wrapped
        if not seed_vectors:
            max_score: float = (
                float(baseline_results[0][2]) if baseline_results else 1.0
            )
            if self.dvs.v(verbose):
                logger.info("[S5] No seed embeddings; fallback to baseline results")
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=None,
                    iterations=0,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score) if max_score > 0 else 0.0,
                )
                for i, (_p, doc, sc) in enumerate(baseline_results[:top_k])
            ]

        from dvs.utils.graph_ops import mean_vector

        context_centroid: list[float] = mean_vector(seed_vectors)
        if self.dvs.v(verbose):
            logger.debug(
                (
                    f"[S5] seed_vecs(after_fetch)={len(seed_vectors)} "
                    + f"centroid_dim={len(context_centroid)}"
                )
            )

        # Query embedding (for vector similarity against candidates)
        query_vector: list[float] = await self.embed_query_vector(query)

        # 2) Single-step graph expansion via entities (is_from)
        #    Optionally respect max_expansion_steps>0; here we perform one step.
        if max_expansion_steps <= 0:
            max_expansion_steps = 1

        entity_ids: set[str] = set()
        for _pt, doc, _ in baseline_results:
            try:
                doc_node = self.dvs.db.graph.nodes.retrieve_by_label(
                    doc.document_id, verbose=False
                )
                neighbors = self.dvs.db.graph.get_neighbors(
                    to_node_id_or_label=doc_node.node_id,
                    relation=RelationIsFrom,
                    limit=200,
                    verbose=False,
                )
                for from_node, _edge, to_node in neighbors:
                    if from_node.kind == "entity":
                        entity_ids.add(from_node.node_id)
                    if to_node.kind == "entity":
                        entity_ids.add(to_node.node_id)
            except Exception:
                continue

        if self.dvs.v(verbose):
            logger.debug(f"[S5] collected_entities={len(entity_ids)}")

        # 2b) Expand via is_a/has_a, then hub suppression via related_to PR
        expanded_entities5: set[str] = set(entity_ids)
        expanded_entities5 = self.expand_is_a_bfs(
            expanded_entities5,
            is_a_max_hops,
            is_a_limit_per_hop,
            cap_total=None,
            verbose=False,
        )
        if has_a_enabled and expanded_entities5:
            expanded_entities5 |= self.expand_has_a_one_hop(
                expanded_entities5,
                has_a_limit_per_entity,
                cap_total=None,
                verbose=False,
            )

        if suppress_hubs and expanded_entities5:
            before_hub5: int = len(expanded_entities5)
            expanded_entities5 = self.suppress_hubs_by_pagerank(
                expanded_entities5,
                hub_pagerank_top_percent,
                relation=RelationRelatedTo,
                limit=5000,
            )
            suppressed5: int = before_hub5 - len(expanded_entities5)
            if self.dvs.v(verbose):
                logger.info(
                    (
                        f"[S5] hub_suppressed={suppressed5} "
                        f"entities_after={len(expanded_entities5)}"
                    )
                )

        # 3) Collect candidate documents from these entities
        candidate_doc_ids: set[str] = self.collect_docs_via_is_from(
            expanded_entities5, per_entity_limit=8, cap_entities=300, verbose=False
        )

        # Remove seeds from candidates
        candidate_doc_ids.difference_update(set(seed_doc_ids))

        if self.dvs.v(verbose):
            logger.debug(
                (
                    f"[S5] candidate_docs={len(candidate_doc_ids)} "
                    + "(after removing seeds)"
                )
            )
            suppressed_val5: int = (
                (before_hub5 - len(expanded_entities5))
                if "before_hub5" in locals()
                else 0
            )
            logger.info(
                (
                    f"[S5] seeds={len(entity_ids)} "
                    f"entities_after={len(expanded_entities5)} "
                    f"suppressed={suppressed_val5} "
                    f"docs_collected={len(candidate_doc_ids)}"
                )
            )

        if not candidate_doc_ids:
            # Fallback: return baseline wrapped
            max_score: float = (
                float(baseline_results[0][2]) if baseline_results else 1.0
            )
            if self.dvs.v(verbose):
                logger.info(
                    "[S5] No candidate docs from entities; fallback to baseline"
                )
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=None,
                    iterations=0,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score) if max_score > 0 else 0.0,
                )
                for i, (_p, doc, sc) in enumerate(baseline_results[:top_k])
            ]

        # 4) Score candidates by query similarity and context similarity
        alpha: float = 0.6  # weight for query-vs-candidate vector similarity
        beta: float = 0.4  # weight for context-vs-candidate similarity

        scored: list[tuple[Document, float, float, float]] = []
        log_counter: int = 0
        for doc_id in list(candidate_doc_ids)[:500]:
            try:
                cand_centroid: list[float] = self.centroid_for_document(doc_id)
                if not cand_centroid:
                    if self.dvs.v(verbose) and log_counter < 10:
                        logger.debug(f"[S5] skip doc={doc_id} (no centroid)")
                    continue
                vec_sim: float = float(cosine_similarity(query_vector, cand_centroid))
                ctx_sim: float = float(
                    cosine_similarity(context_centroid, cand_centroid)
                )
                if ctx_sim < float(context_similarity_threshold):
                    if self.dvs.v(verbose) and log_counter < 10:
                        logger.debug(
                            (
                                f"[S5] drop doc={doc_id} vec={vec_sim:.3f} "
                                + f"ctx={ctx_sim:.3f} "
                                + f"th={context_similarity_threshold:.2f}"
                            )
                        )
                        log_counter += 1
                    continue
                combined: float = alpha * vec_sim + beta * ctx_sim
                doc = self.dvs.db.documents.retrieve(doc_id, verbose=False)
                scored.append((doc, combined, vec_sim, ctx_sim))
                if self.dvs.v(verbose) and log_counter < 10:
                    logger.debug(
                        (
                            f"[S5] keep doc={doc_id} vec={vec_sim:.3f} "
                            + f"ctx={ctx_sim:.3f} combined={combined:.3f}"
                        )
                    )
                    log_counter += 1
            except Exception:
                continue

        if not scored:
            # Fallback: return baseline wrapped
            max_score: float = (
                float(baseline_results[0][2]) if baseline_results else 1.0
            )
            if self.dvs.v(verbose):
                logger.info(
                    "[S5] No candidates passed context threshold; fallback to baseline"
                )
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=None,
                    iterations=0,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score) if max_score > 0 else 0.0,
                )
                for i, (_p, doc, sc) in enumerate(baseline_results[:top_k])
            ]

        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        max_score: float = float(top[0][1]) if top else 1.0
        out: list[GraphRAGResult] = []
        for idx, (doc, combined, vec_sim, ctx_sim) in enumerate(top, start=1):
            norm: float = (float(combined) / max_score) if max_score > 0 else 0.0
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=float(combined),
                    vector_score=float(vec_sim),
                    graph_score=float(ctx_sim),
                    iterations=1,
                    rank=idx,
                    normalized_score=norm,
                )
            )
        return out

    def calculate_graph_relevance(
        self,
        target_doc_id: str,
        original_doc_ids: list[str],
        related_entities: set[str],
        *,
        verbose: bool | None = None,
    ) -> float:
        """Calculate graph relevance for a document"""
        # Give higher score if document is in original search results
        if target_doc_id in original_doc_ids:
            return 1.0

        # Calculate graph distance to original documents
        min_distance = float("inf")

        try:
            # Try to find target document node
            # Resolve by label (document_id)
            target_node = self.dvs.db.graph.nodes.retrieve_by_label(
                target_doc_id, verbose=self.dvs.v(verbose)
            )

            for original_doc_id in original_doc_ids:
                try:
                    original_node = self.dvs.db.graph.nodes.retrieve_by_label(
                        original_doc_id, verbose=False
                    )

                    # Calculate shortest path distance
                    paths = self.dvs.db.graph.get_shortest_paths(
                        from_node_id_or_label=original_node.node_id,
                        to_node_id_or_label=target_node.node_id,
                        limit=1,
                        verbose=False,
                    )

                    if paths:
                        distance = paths[0][2]  # Distance is in third position
                        min_distance = min(min_distance, distance)

                except Exception:
                    continue

        except Exception:
            return 0.1  # Default low score

        # Higher score for closer distance
        if min_distance == float("inf"):
            return 0.1
        elif min_distance == 0:
            return 1.0
        else:
            return max(0.1, 1.0 / (min_distance + 1))
