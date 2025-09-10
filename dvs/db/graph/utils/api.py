# dvs/db/graph/utils/api.py
import concurrent.futures
import logging
import textwrap
import typing
from concurrent.futures import ThreadPoolExecutor

import duckdb
import pydantic

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
from dvs.types.node import Node as NodeType
from dvs.types.point import Point
from dvs.utils.debug_print import debug_print
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


DEFAULT_IS_A_HOPS = 10
DEFAULT_IS_A_PER_HOP_LIMIT = 20
DEFAULT_IS_A_MAX_ENTITIES = 100
DEFAULT_HAS_A_HOPS = 1
DEFAULT_HAS_A_PER_HOP_LIMIT = 20
DEFAULT_HAS_A_MAX_ENTITIES = 100
DEFAULT_RELATED_TO_HOPS = 1
DEFAULT_RELATED_TO_PER_HOP_LIMIT = 20
DEFAULT_RELATED_TO_MAX_ENTITIES = 100


class Utils:
    def __init__(self, dvs: dvs.DVS):
        """Graph utils API"""
        self.dvs = dvs

    def get_neighbors(
        self,
        from_: str | None = None,  # From node ID
        to: str | None = None,  # To node ID
        *,
        any_direction: bool = False,
        from_kind: typing.Literal["entity", "document"] | None = None,
        to_kind: typing.Literal["entity", "document"] | None = None,
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
            from_clauses = []
            to_clauses = []
            if from_ is not None:
                from_clauses.append(f"{FROM_NODE_ALIAS}.node_id = '{from_}'")
            if to is not None:
                to_clauses.append(f"{TO_NODE_ALIAS}.node_id = '{to}'")
            if from_kind is not None:
                from_clauses.append(f"{FROM_NODE_ALIAS}.kind = '{from_kind}'")
            if to_kind is not None:
                to_clauses.append(f"{TO_NODE_ALIAS}.kind = '{to_kind}'")
            from_clause = (
                (" WHERE " + " AND ".join(from_clauses)) if from_clauses else ""
            )
            to_clause = (" WHERE " + " AND ".join(to_clauses)) if to_clauses else ""

            queries = [
                textwrap.dedent(
                    f"""
                    FROM GRAPH_TABLE (
                        {dvs.DVS_GRAPH_TABLE_NAME}
                        MATCH
                            (
                                {FROM_NODE_ALIAS}:nodes{from_clause}
                            )-[{RELATION_ALIAS}:{query_relation}]{'-' if any_direction is True else '->'}(
                                {TO_NODE_ALIAS}:nodes{to_clause}
                            )
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

    def walk_neighbors(
        self,
        seed_entities: typing.List[NodeType],
        *,
        relation: RelationType,
        max_hops: int,
        per_hop_limit: int,
        max_entities: int,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> typing.List[NodeType]:
        """Walk neighbors of entities."""
        if max_hops <= 0 or not seed_entities:
            return seed_entities

        expanded: set[NodeType] = set(seed_entities)
        frontier: set[NodeType] = set(seed_entities)
        visited: set[NodeType] = set(seed_entities)

        for _ in range(max_hops):
            if not frontier:
                break
            next_frontier: set[NodeType] = set()
            for entity in list(frontier):
                neighbors = self.get_neighbors(
                    from_=entity.node_id,
                    relation=relation,
                    to_kind="entity",
                    any_direction=True,
                    limit=per_hop_limit,
                    conn=conn,
                    verbose=self.dvs.v(verbose),
                )

                for _, _, to_node in neighbors:
                    if to_node not in visited:
                        visited.add(to_node)
                        expanded.add(to_node)
                        next_frontier.add(to_node)
                        if max_entities is not None and len(expanded) >= max_entities:
                            return list(expanded)
            frontier = next_frontier

        return list(expanded)

    def expand_entity_nodes(
        self,
        seed_entities: typing.List[NodeType],
        *,
        enable_is_a: bool = True,
        max_hops_is_a: int = DEFAULT_IS_A_HOPS,
        per_hop_limit_is_a: int = DEFAULT_IS_A_PER_HOP_LIMIT,
        max_entities_is_a: int = DEFAULT_IS_A_MAX_ENTITIES,
        enable_has_a: bool = True,
        max_hops_has_a: int = DEFAULT_HAS_A_HOPS,
        per_hop_limit_has_a: int = DEFAULT_HAS_A_PER_HOP_LIMIT,
        max_entities_has_a: int = DEFAULT_HAS_A_MAX_ENTITIES,
        enable_related_to: bool = False,
        max_hops_related_to: int = DEFAULT_RELATED_TO_HOPS,
        per_hop_limit_related_to: int = DEFAULT_RELATED_TO_PER_HOP_LIMIT,
        max_entities_related_to: int = DEFAULT_RELATED_TO_MAX_ENTITIES,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> typing.List[NodeType]:
        """Expand entities."""
        if not seed_entities:
            return []

        NodesList = pydantic.TypeAdapter(typing.List[NodeType])
        expanded_entities: set[NodeType] = set()

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures: list[concurrent.futures.Future[typing.List[NodeType]]] = []
            if enable_is_a:
                futures.append(
                    executor.submit(
                        self.walk_neighbors,
                        NodesList.validate_json(NodesList.dump_json(seed_entities)),
                        relation=RelationIsA,
                        max_hops=max_hops_is_a,
                        per_hop_limit=per_hop_limit_is_a,
                        max_entities=max_entities_is_a,
                        conn=conn,
                        verbose=self.dvs.v(verbose),
                    )
                )
            if enable_has_a:
                futures.append(
                    executor.submit(
                        self.walk_neighbors,
                        NodesList.validate_json(NodesList.dump_json(seed_entities)),
                        relation=RelationHasA,
                        max_hops=max_hops_has_a,
                        per_hop_limit=per_hop_limit_has_a,
                        max_entities=max_entities_has_a,
                        conn=conn,
                        verbose=self.dvs.v(verbose),
                    )
                )
            if enable_related_to:
                futures.append(
                    executor.submit(
                        self.walk_neighbors,
                        NodesList.validate_json(NodesList.dump_json(seed_entities)),
                        relation=RelationRelatedTo,
                        max_hops=max_hops_related_to,
                        per_hop_limit=per_hop_limit_related_to,
                        max_entities=max_entities_related_to,
                        conn=conn,
                        verbose=self.dvs.v(verbose),
                    )
                )
            for future in futures:
                expanded_entities.update(future.result())

        return list(expanded_entities)

    def get_document_nodes_from_entities(
        self,
        entities: typing.List[NodeType] | typing.List[str],
        per_entity_limit: int = 10,
        *,
        max_documents: int | None = None,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> typing.List[NodeType]:
        """Collect document ids reachable via is_from from entities."""
        collected: typing.Set[NodeType] = set()
        if not entities:
            return list(collected)

        for entity in entities:
            neighbors = self.get_neighbors(
                from_=entity if isinstance(entity, str) else entity.node_id,
                relation=RelationIsFrom,
                to_kind="document",
                limit=per_entity_limit,
                conn=conn,
                verbose=self.dvs.v(verbose),
            )
            for _, _, to_node in neighbors:
                collected.add(to_node)
        return list(collected)

    def get_entity_nodes_from_documents(
        self,
        documents: typing.List[NodeType] | typing.List[str],
        per_document_limit: int = 10,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> typing.List[NodeType]:
        """Collect entity nodes from documents."""
        collected: typing.Set[NodeType] = set()
        if not documents:
            return list(collected)
        for document in documents:
            neighbors = self.get_neighbors(
                to=document if isinstance(document, str) else document.node_id,
                relation=RelationIsFrom,
                from_kind="entity",
                limit=per_document_limit,
                conn=conn,
                verbose=self.dvs.v(verbose),
            )
            for _, _, from_node in neighbors:
                collected.add(from_node)
        return list(collected)

    def centroid_for_document(
        self,
        doc: NodeType | str,
        *,
        limit_points: int = 5,
        conn: duckdb.DuckDBPyConnection | None = None,
    ) -> list[float]:
        """Compute centroid from a document's point embeddings."""
        from dvs.utils.graph_ops import mean_vector

        vectors: list[list[float]] = []
        pts = self.dvs.db.points.gen(
            document_id=(
                self.get_node_by_id_with_cache(doc).label
                if isinstance(doc, str)
                else doc.label
            ),
            limit=limit_points,
            with_embedding=True,
            conn=conn,
            verbose=False,
        )
        for p in pts:
            if p.embedding:
                vectors.append(p.to_python())

        return mean_vector(vectors)

    def points_for_documents(
        self,
        documents: typing.List[NodeType] | typing.List[str],
        *,
        per_doc_limit: int = 10,
        with_embedding: bool = True,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> list[Point]:
        """Gather points for documents; optionally include embeddings."""
        points: list[Point] = []
        for pt in self.dvs.db.points.gen(
            document_ids=[
                (
                    self.get_node_by_id_with_cache(doc).label
                    if isinstance(doc, str)
                    else doc.label
                )
                for doc in documents
            ],
            limit=per_doc_limit,
            with_embedding=with_embedding,
            conn=conn,
            verbose=self.dvs.v(verbose),
        ):
            points.append(pt)
        return points

    def get_node_by_id_with_cache(
        self,
        node_id: str,
        *,
        cache: dict[str, "NodeType"] | None = None,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> "NodeType":
        """Get node from cache or retrieve from database and cache it."""
        if cache is None:
            cache = {}
        if node_id in cache:
            return cache[node_id]
        node = self.dvs.db.graph.nodes.retrieve_or_raise(
            node_id, conn=conn, verbose=self.dvs.v(verbose)
        )
        cache[node_id] = node
        return node

    def calculate_graph_relevance(
        self,
        target_doc_id: str,
        original_doc_ids: list[str],
        related_entities: list[NodeType],
        *,
        doc_ids_nodes: dict[str, "NodeType"] | None = None,
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

        # Create document id to node cache to avoid repeated database calls
        doc_ids_nodes = {} if doc_ids_nodes is None else doc_ids_nodes

        # If no related entities provided, fall back to distance-based scoring
        if not related_entities:
            return self.calculate_distance_based_relevance(
                target_doc_id,
                original_doc_ids,
                doc_ids_nodes,
                conn=conn,
                verbose=verbose,
            )

        # Calculate entity overlap score (primary algorithm)
        entity_overlap_score = self.calculate_entity_overlap_score(
            target_doc_id, related_entities, doc_ids_nodes, conn=conn, verbose=verbose
        )

        # Calculate distance score as fallback/supplement
        distance_score = self.calculate_distance_based_relevance(
            target_doc_id, original_doc_ids, doc_ids_nodes, conn=conn, verbose=verbose
        )

        # Combine scores: prioritize entity overlap but include distance as baseline
        # Entity overlap is more direct indicator of relevance in our expanded graph
        combined_score = 0.8 * entity_overlap_score + 0.2 * distance_score

        return max(0.1, min(1.0, combined_score))

    def calculate_entity_overlap_score(
        self,
        target_doc: NodeType | str,
        related_entities: typing.List[NodeType] | typing.List[str],
        doc_ids_nodes: dict[str, "NodeType"],
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> float:
        """
        Calculate relevance based on entity overlap with expanded entity set.

        Args:
            target_doc_id: Target document ID
            related_entities: Set of expanded entities from graph expansion
            doc_ids_nodes: Cached mapping of document IDs to Node objects

        Returns:
            Relevance score [0,1] based on entity overlap
        """
        if not related_entities:
            return 0.0

        try:
            # Use cached node or retrieve if not available
            target_node = self.get_node_by_id_with_cache(
                target_doc if isinstance(target_doc, str) else target_doc.node_id,
                cache=doc_ids_nodes,
                conn=conn,
                verbose=verbose,
            )

            # Find entities connected to this document via is_from relationship
            doc_entities: set[str] = set()
            neighbors = self.get_neighbors(
                to=target_node.node_id,
                relation=RelationIsFrom,
                from_kind="entity",
                limit=50,  # Get reasonable number of connected entities
                conn=conn,
                verbose=False,
            )

            for _, _, from_node in neighbors:
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

    def calculate_distance_based_relevance(
        self,
        target_doc_id: str,
        original_doc_ids: list[str],
        doc_ids_nodes: dict[str, "NodeType"],
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> float:
        """
        Calculate relevance based on graph distance to original documents.

        Args:
            target_doc_id: Target document ID
            original_doc_ids: List of original seed document IDs
            doc_ids_nodes: Cached mapping of document IDs to Node objects

        Returns:
            Relevance score [0,1] based on graph distance
        """
        # Calculate graph distance to original documents
        min_distance = float("inf")

        try:
            # Use cached target node or retrieve if not available
            target_node = self.get_node_by_id_with_cache(
                target_doc_id, cache=doc_ids_nodes, conn=conn, verbose=verbose
            )

            for original_doc_id in original_doc_ids:
                try:
                    # Use cached original node or retrieve if not available
                    original_node = self.get_node_by_id_with_cache(
                        original_doc_id, cache=doc_ids_nodes, conn=conn, verbose=False
                    )

                    # Calculate shortest path distance
                    paths = self.dvs.db.graph.algorithm.get_shortest_paths(
                        from_=original_node.node_id,
                        to=target_node.node_id,
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
        conn = conn or self.dvs.new_connection()

        # Vector search to find seed documents
        initial_results = await self.dvs.search(
            query=query,
            top_k=top_k * 2,
            with_embedding=True,
            conn=conn,
            verbose=self.dvs.v(verbose),
        )

        if not initial_results:
            return []

        document_ids = [doc.document_id for _, doc, _ in initial_results]

        # Entity discovery & graph expansion
        related_entities: set[NodeType] = set()
        if document_ids:
            related_entities.update(
                self.get_entity_nodes_from_documents(
                    document_ids, per_document_limit=10, conn=conn, verbose=False
                )
            )

        # Expand entities through multiple relations
        expanded_entities = self.expand_entity_nodes(
            list(related_entities),
            enable_is_a=True,
            max_hops_is_a=is_a_max_hops,
            per_hop_limit_is_a=is_a_limit_per_hop,
            max_entities_is_a=entity_expansion_cap,
            enable_has_a=has_a_enabled,
            max_hops_has_a=has_a_limit_per_entity,
            per_hop_limit_has_a=has_a_limit_per_entity,
            max_entities_has_a=entity_expansion_cap,
            enable_related_to=related_to_enabled,
            max_hops_related_to=related_to_limit_per_entity,
            per_hop_limit_related_to=related_to_limit_per_entity,
            max_entities_related_to=entity_expansion_cap,
            conn=conn,
            verbose=verbose,
        )

        # Cache document nodes
        doc_ids_nodes: dict[str, "NodeType"] = {}
        all_doc_ids = set(document_ids)
        for doc_node in self.dvs.db.graph.nodes.retrieve_by_ids(
            list(all_doc_ids), conn=conn, verbose=False
        ):
            if doc_node.node_id not in doc_ids_nodes:
                doc_ids_nodes[doc_node.node_id] = doc_node

        # Calculate combined scores
        enhanced_results: list[tuple["Point", "Document", float]] = []
        for point, doc, vec_score in initial_results:
            try:
                graph_score: float = self.calculate_graph_relevance(
                    doc.document_id,
                    document_ids,
                    list(expanded_entities),
                    doc_ids_nodes=doc_ids_nodes,
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
