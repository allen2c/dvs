# dvs/db/graph/utils/api.py
import logging
import textwrap
import typing
from concurrent.futures import ThreadPoolExecutor

import duckdb

import dvs
from dvs.types.document import Document
from dvs.types.edge import Edge as EdgeType
from dvs.types.edge import (
    RelationHasA,
    RelationIsA,
    RelationIsFrom,
    RelationRelatedTo,
    RelationType,
)
from dvs.types.graphrag_result import GraphRAGResult
from dvs.types.node import Node as NodeType
from dvs.types.point import Point
from dvs.utils.debug_print import debug_print
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Utils:
    def __init__(self, dvs: dvs.DVS):
        """Graph utils API"""
        self.dvs = dvs

    def get_neighbors(
        self,
        from_node_id_or_label: str | None = None,
        to_node_id_or_label: str | None = None,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
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
        conn = conn or self.dvs.new_connection()

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

    def expand_is_a_bfs(
        self,
        seed_entities: set[str],
        max_hops: int,
        per_hop_limit: int,
        *,
        cap_total: int | None = None,
        conn: duckdb.DuckDBPyConnection | None = None,
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
                        conn=conn,
                        verbose=self.dvs.v(verbose),
                    )
                except Exception:
                    outs = []
                try:
                    ins = self.get_neighbors(
                        to_node_id_or_label=entity_id,
                        relation=RelationIsA,
                        limit=per_hop_limit,
                        conn=conn,
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
        conn: duckdb.DuckDBPyConnection | None = None,
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
                    conn=conn,
                    verbose=self.dvs.v(verbose),
                )
            except Exception:
                outs = []
            try:
                ins = self.get_neighbors(
                    to_node_id_or_label=eid,
                    relation=RelationHasA,
                    limit=per_entity_limit,
                    conn=conn,
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
        conn: duckdb.DuckDBPyConnection | None = None,
        relation: RelationType = RelationRelatedTo,
        limit: int = 5000,
    ) -> set[str]:
        """Remove high-PageRank hubs above the given percentile cutoff."""
        if not entity_ids:
            return set()
        try:
            pr = self.dvs.db.graph.algorithm.pagerank(
                conn=conn,
                relation=relation,
                limit=limit,
                verbose=False,
            )
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
        conn: duckdb.DuckDBPyConnection | None = None,
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
                    conn=conn,
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
        self,
        doc_id: str,
        *,
        limit_points: int = 5,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> list[float]:
        """Compute centroid from a document's point embeddings."""
        vectors: list[list[float]] = []
        try:
            pts = self.dvs.db.points.gen(
                document_id=doc_id,
                limit=limit_points,
                with_embedding=True,
                conn=conn,
                verbose=False,
            )
            for p in pts:
                if p.embedding:
                    vectors.append(p.to_python())
        except Exception:
            pass
        from dvs.utils.graph_ops import mean_vector

        return mean_vector(vectors)

    def collect_entities_via_is_from_for_documents(
        self,
        document_ids: list[str],
        *,
        limit_per_doc: int = 10,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> set[str]:
        """Collect entities connected to documents via is_from edges."""
        entity_ids: set[str] = set()
        for doc_node in self.dvs.db.graph.nodes.retrieve_by_labels(
            document_ids, conn=conn, verbose=self.dvs.v(verbose)
        ):
            neighbors = self.get_neighbors(
                to_node_id_or_label=doc_node.node_id,
                relation=RelationIsFrom,
                limit=limit_per_doc,
                conn=conn,
                verbose=self.dvs.v(verbose),
            )
            for from_node, _edge, to_node in neighbors:
                if getattr(from_node, "kind", None) == "entity":
                    entity_ids.add(from_node.node_id)
                if getattr(to_node, "kind", None) == "entity":
                    entity_ids.add(to_node.node_id)

        return entity_ids

    def gather_points_for_documents(
        self,
        document_ids: list[str] | set[str],
        *,
        per_doc_limit: int = 10,
        with_embedding: bool = True,
        conn: duckdb.DuckDBPyConnection | None = None,
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
                    conn=conn,
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
        conn: duckdb.DuckDBPyConnection | None = None,
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
                conn=conn,
                verbose=self.dvs.v(verbose),
            )
        if has_a_enabled and expanded:
            expanded |= self.expand_has_a_one_hop(
                expanded,
                has_a_limit_per_entity,
                cap_total=cap_total,
                conn=conn,
                verbose=self.dvs.v(verbose),
            )
        if suppress_hubs and expanded:
            expanded = self.suppress_hubs_by_pagerank(
                expanded,
                hub_pagerank_top_percent,
                conn=conn,
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
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> float:
        """Return normalized graph importance [0,1] for a document node."""
        if document_id in pagerank_map and max_pagerank > 0:
            return pagerank_map[document_id] / max_pagerank

        try:
            neighbors = self.get_neighbors(
                to_node_id_or_label=document_id,
                relation=RelationIsFrom,
                limit=30,
                conn=conn,
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
                    conn=conn,
                    verbose=False,
                )

            if has_a_enabled and entities:
                entities |= self.expand_has_a_one_hop(
                    entities,
                    has_a_limit_per_entity,
                    conn=conn,
                    verbose=False,
                )

            if suppress_hubs and entities:
                entities = self.suppress_hubs_by_pagerank(
                    entities,
                    hub_pagerank_top_percent,
                    conn=conn,
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
        self,
        document_id: str,
        original_doc_ids: list[str],
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> float:
        """Distance score [0,1] derived from shortest path to seeds."""
        return self.calculate_graph_relevance(
            document_id,
            original_doc_ids,
            related_entities=set(),
            conn=conn,
            verbose=False,
        )

    def calculate_graph_relevance(
        self,
        target_doc_id: str,
        original_doc_ids: list[str],
        related_entities: set[str],
        *,
        doc_label_nodes: dict[str, "NodeType"] | None = None,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> float:
        """
        Calculate graph relevance for a document using entity-based scoring.

        Algorithm combines:
        1. Entity overlap score: How many expanded entities does this
           document connect to?
        2. Distance score: Graph distance to original documents (fallback)

        Args:
            target_doc_id: Target document ID to calculate relevance for
            original_doc_ids: List of original seed document IDs
            related_entities: Set of expanded entities from graph expansion
            conn: Database connection (optional)
            verbose: Verbosity flag (optional)

        Returns:
            Relevance score [0,1] combining entity overlap and distance
        """
        # Give maximum score if document is in original search results
        if target_doc_id in original_doc_ids:
            return 1.0

        # Create document label to node cache to avoid repeated database calls
        doc_label_nodes = {} if doc_label_nodes is None else doc_label_nodes

        # If no related entities provided, fall back to distance-based scoring
        if not related_entities:
            return self.calculate_distance_based_relevance(
                target_doc_id,
                original_doc_ids,
                doc_label_nodes,
                conn=conn,
                verbose=verbose,
            )

        # Calculate entity overlap score (primary algorithm)
        entity_overlap_score = self.calculate_entity_overlap_score(
            target_doc_id, related_entities, doc_label_nodes, conn=conn, verbose=verbose
        )

        # Calculate distance score as fallback/supplement
        distance_score = self.calculate_distance_based_relevance(
            target_doc_id, original_doc_ids, doc_label_nodes, conn=conn, verbose=verbose
        )

        # Combine scores: prioritize entity overlap but include distance as baseline
        # Entity overlap is more direct indicator of relevance in our expanded graph
        combined_score = 0.8 * entity_overlap_score + 0.2 * distance_score

        return max(0.1, min(1.0, combined_score))

    def calculate_entity_overlap_score(
        self,
        target_doc_id: str,
        related_entities: set[str],
        doc_label_nodes: dict[str, "NodeType"],
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> float:
        """
        Calculate relevance based on entity overlap with expanded entity set.

        Args:
            target_doc_id: Target document ID
            related_entities: Set of expanded entities from graph expansion
            doc_label_nodes: Cached mapping of document IDs to Node objects

        Returns:
            Relevance score [0,1] based on entity overlap
        """
        if not related_entities:
            return 0.0

        try:
            # Use cached node or retrieve if not available
            if target_doc_id in doc_label_nodes:
                target_node = doc_label_nodes[target_doc_id]
            else:
                target_node = self.dvs.db.graph.nodes.retrieve_by_label_or_raise(
                    target_doc_id, conn=conn, verbose=self.dvs.v(verbose)
                )
                doc_label_nodes[target_doc_id] = target_node

            # Find entities connected to this document via is_from relationship
            doc_entities: set[str] = set()
            neighbors = self.get_neighbors(
                to_node_id_or_label=target_node.node_id,
                relation=RelationIsFrom,
                limit=50,  # Get reasonable number of connected entities
                conn=conn,
                verbose=False,
            )

            for _, _, from_node in neighbors:
                if getattr(from_node, "kind", None) == "entity":
                    doc_entities.add(from_node.node_id)

            if not doc_entities:
                return 0.1  # Document has no connected entities

            # Calculate overlap ratio
            overlap_count = len(doc_entities.intersection(related_entities))
            overlap_ratio = (
                overlap_count / len(related_entities) if related_entities else 0.0
            )

            # Add small bonus for any overlap (even if small ratio)
            overlap_bonus = min(0.2, overlap_count * 0.1)

            # Final score: combination of ratio and bonus
            score = overlap_ratio + overlap_bonus

            return min(1.0, score)

        except Exception:
            return 0.1  # Default low score on error

    def expand_entities_with_multiple_relations(
        self,
        seed_entities: set[str],
        *,
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        related_to_enabled: bool = True,
        related_to_limit_per_entity: int = 1,
        suppress_hubs: bool = False,
        hub_pagerank_top_percent: float = 0.1,
        entity_expansion_cap: int = 200,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> set[str]:
        """
        Expand entities through multiple relation types (is_a, has_a, related_to).

        This consolidates the common expansion pattern used in graph search strategies.
        Includes optional hub suppression using PageRank.

        Args:
            seed_entities: Initial set of entity IDs to expand from
            is_a_max_hops: Maximum hops for is_a expansion (multi-hop BFS)
            is_a_limit_per_hop: Limit per hop for is_a expansion
            has_a_enabled: Whether to enable has_a expansion
            has_a_limit_per_entity: Limit per entity for has_a expansion
            related_to_enabled: Whether to enable related_to expansion
            related_to_limit_per_entity: Limit per entity for related_to expansion
            suppress_hubs: Whether to suppress high-PageRank hub entities
            hub_pagerank_top_percent: Top percentile to suppress for hub removal
            entity_expansion_cap: Maximum total entities to expand to
            conn: Database connection
            verbose: Verbosity flag

        Returns:
            Expanded set of entity IDs
        """
        expanded_entity_ids: set[str] = set(seed_entities)

        # is_a expansion (multi-hop BFS)
        if is_a_max_hops > 0 and len(expanded_entity_ids) < entity_expansion_cap:
            expanded_entity_ids = self.expand_is_a_bfs(
                expanded_entity_ids,
                is_a_max_hops,
                is_a_limit_per_hop,
                cap_total=entity_expansion_cap,
                conn=conn,
                verbose=verbose,
            )

        # has_a expansion (1-hop)
        if has_a_enabled and len(expanded_entity_ids) < entity_expansion_cap:
            expanded_entity_ids |= self.expand_has_a_one_hop(
                expanded_entity_ids,
                has_a_limit_per_entity,
                cap_total=entity_expansion_cap,
                conn=conn,
                verbose=verbose,
            )

        # related_to expansion (1-hop)
        if related_to_enabled and len(expanded_entity_ids) < entity_expansion_cap:
            for eid in list(expanded_entity_ids)[:entity_expansion_cap]:
                try:
                    out_neighbors = self.get_neighbors(
                        from_node_id_or_label=eid,
                        relation=RelationRelatedTo,
                        limit=related_to_limit_per_entity,
                        conn=conn,
                        verbose=verbose,
                    )
                    in_neighbors = self.get_neighbors(
                        to_node_id_or_label=eid,
                        relation=RelationRelatedTo,
                        limit=related_to_limit_per_entity,
                        conn=conn,
                        verbose=verbose,
                    )

                    for from_node, _edge, to_node in list(out_neighbors) + list(
                        in_neighbors
                    ):
                        other = to_node if from_node.node_id == eid else from_node
                        if getattr(other, "kind", None) == "entity":
                            expanded_entity_ids.add(other.node_id)
                            if len(expanded_entity_ids) >= entity_expansion_cap:
                                break
                    if len(expanded_entity_ids) >= entity_expansion_cap:
                        break
                except Exception:
                    continue

        # Optional hub suppression using PageRank
        if suppress_hubs and expanded_entity_ids:
            try:
                expanded_entity_ids = self.suppress_hubs_by_pagerank(
                    expanded_entity_ids,
                    hub_pagerank_top_percent,
                    conn=conn,
                    relation=RelationRelatedTo,
                    limit=5000,
                )
            except Exception:
                pass  # Continue without suppression if it fails

        return expanded_entity_ids

    def calculate_distance_based_relevance(
        self,
        target_doc_id: str,
        original_doc_ids: list[str],
        doc_label_nodes: dict[str, "NodeType"],
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> float:
        """
        Calculate relevance based on graph distance to original documents.

        Args:
            target_doc_id: Target document ID
            original_doc_ids: List of original seed document IDs
            doc_label_nodes: Cached mapping of document IDs to Node objects

        Returns:
            Relevance score [0,1] based on graph distance
        """
        # Calculate graph distance to original documents
        min_distance = float("inf")

        try:
            # Use cached target node or retrieve if not available
            if target_doc_id in doc_label_nodes:
                target_node = doc_label_nodes[target_doc_id]
            else:
                target_node = self.dvs.db.graph.nodes.retrieve_by_label_or_raise(
                    target_doc_id, conn=conn, verbose=self.dvs.v(verbose)
                )
                doc_label_nodes[target_doc_id] = target_node

            for original_doc_id in original_doc_ids:
                try:
                    # Use cached original node or retrieve if not available
                    if original_doc_id in doc_label_nodes:
                        original_node = doc_label_nodes[original_doc_id]
                    else:
                        original_node = (
                            self.dvs.db.graph.nodes.retrieve_by_label_or_raise(
                                original_doc_id, conn=conn, verbose=False
                            )
                        )
                        doc_label_nodes[original_doc_id] = original_node

                    # Calculate shortest path distance
                    paths = self.dvs.db.graph.algorithm.get_shortest_paths(
                        from_node_id_or_label=original_node.node_id,
                        to_node_id_or_label=target_node.node_id,
                        limit=1,
                        conn=conn,
                        verbose=False,
                    )

                    if paths:
                        distance = paths[0][2]  # Distance is in third position
                        min_distance = min(min_distance, distance)

                except Exception:
                    continue

        except Exception:
            return 0.1  # Default low score

        # Convert distance to relevance score
        if min_distance == float("inf"):
            return 0.1
        elif min_distance == 0:
            return 1.0
        else:
            return max(0.1, 1.0 / (min_distance + 1))

    async def perform_vector_expansion_search(
        self,
        dvs: "dvs.DVS",
        query: str,
        top_k: int,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        is_a_max_hops: int = 3,
        is_a_limit_per_hop: int = 20,
        has_a_enabled: bool = True,
        has_a_limit_per_entity: int = 3,
        related_to_enabled: bool = True,
        related_to_limit_per_entity: int = 1,
        entity_expansion_cap: int = 200,
        verbose: bool | None = None,
    ) -> list[tuple["Point", "Document", float]]:
        """
        Core vector expansion logic for GraphRAG search strategies.

        Performs vector search, entity discovery, graph expansion, and
        weighted scoring combination. Used as a building block for
        various search strategies.

        Args:
            dvs: DVS instance for database access and utilities
            query: Search query string
            top_k: Number of top results to return
            conn: Database connection (optional)
            vector_weight: Weight for vector similarity in scoring (0.0-1.0)
            graph_weight: Weight for graph relevance in scoring (0.0-1.0)
            is_a_max_hops: Maximum hops for is_a expansion
            is_a_limit_per_hop: Limit per hop for is_a expansion
            has_a_enabled: Whether to enable has_a expansion
            has_a_limit_per_entity: Limit per entity for has_a expansion
            related_to_enabled: Whether to enable related_to expansion
            related_to_limit_per_entity: Limit per entity for related_to expansion
            entity_expansion_cap: Maximum entities to expand to
            verbose: Verbosity flag

        Returns:
            List of tuples: (Point, Document, combined_score)
        """
        conn = conn or dvs.new_connection()

        # Vector search to find seed documents
        initial_results = await dvs.search(
            query=query,
            top_k=top_k * 2,
            with_embedding=True,
            conn=conn,
            verbose=dvs.v(verbose),
        )

        if not initial_results:
            return []

        document_ids = [doc.document_id for _, doc, _ in initial_results]

        # Entity discovery & graph expansion
        related_entities: set[str] = set()
        if document_ids:
            related_entities = self.collect_entities_via_is_from_for_documents(
                document_ids, limit_per_doc=10, conn=conn, verbose=False
            )

        # Expand entities through multiple relations
        expanded_entity_ids = self.expand_entities_with_multiple_relations(
            related_entities,
            is_a_max_hops=is_a_max_hops,
            is_a_limit_per_hop=is_a_limit_per_hop,
            has_a_enabled=has_a_enabled,
            has_a_limit_per_entity=has_a_limit_per_entity,
            related_to_enabled=related_to_enabled,
            related_to_limit_per_entity=related_to_limit_per_entity,
            suppress_hubs=False,
            entity_expansion_cap=entity_expansion_cap,
            conn=conn,
            verbose=verbose,
        )

        # Cache document nodes
        doc_label_nodes: dict[str, "NodeType"] = {}
        all_doc_ids = set(document_ids)
        for doc_node in dvs.db.graph.nodes.retrieve_by_labels(
            list(all_doc_ids), conn=conn, verbose=False
        ):
            if doc_node.node_id not in doc_label_nodes:
                doc_label_nodes[doc_node.node_id] = doc_node

        # Calculate combined scores
        enhanced_results: list[tuple["Point", "Document", float]] = []
        for point, doc, vec_score in initial_results:
            try:
                graph_score: float = self.calculate_graph_relevance(
                    doc.document_id,
                    document_ids,
                    expanded_entity_ids,
                    doc_label_nodes=doc_label_nodes,
                    conn=conn,
                    verbose=False,
                )
                combined_score: float = (
                    vector_weight * float(vec_score) + graph_weight * graph_score
                )
                enhanced_results.append((point, doc, combined_score))
            except Exception:
                enhanced_results.append((point, doc, float(vec_score)))

        return enhanced_results
