import asyncio
import functools
import logging
import pathlib
import textwrap
import time
import typing

import duckdb
from agents import OpenAIChatCompletionsModel, OpenAIResponsesModel

import dvs
import dvs.utils.vss as VSS
from dvs.types.document import Document
from dvs.types.edge import (
    RelationHasA,
    RelationIsA,
    RelationIsFrom,
    RelationRelatedTo,
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

    from dvs.db.graph.algorithm.api import Algorithm
    from dvs.db.graph.edges.api import Edges
    from dvs.db.graph.nodes.api import Nodes
    from dvs.db.graph.utils.api import Utils

logger = logging.getLogger(__name__)


class Graph:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(
        self,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> bool:
        """Initialize the property graph with required tables and extensions."""
        self.nodes.touch(verbose=self.dvs.v(verbose))
        self.edges.touch(verbose=self.dvs.v(verbose))

        with Timer() as timer:
            conn = conn or self.dvs.new_connection()
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

    def drop(
        self,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        verbose: bool | None = None,
    ) -> bool:
        self.nodes.drop(verbose=self.dvs.v(verbose))
        self.edges.drop(verbose=self.dvs.v(verbose))
        with Timer() as timer:
            conn = conn or self.dvs.new_connection()

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
        conn: duckdb.DuckDBPyConnection | None = None,
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

        self.drop(conn=conn, verbose=self.dvs.v(verbose))
        self.touch(conn=conn, verbose=self.dvs.v(verbose))

        documents = [
            doc
            for doc in self.dvs.db.documents.gen(conn=conn, verbose=self.dvs.v(verbose))
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
        self.nodes.bulk_create(nodes, conn=conn)
        self.edges.bulk_create(edges, conn=conn)

        logger.info("✨ Knowledge Graph Construction Complete! ✨")
        logger.info(f"Total Nodes: {len(nodes)}, Edges: {len(edges)}")
        return G

    def to_nx(self, *, conn: duckdb.DuckDBPyConnection | None = None) -> "nx.DiGraph":
        import networkx as nx

        G = nx.DiGraph()
        for __node in self.nodes.gen(conn=conn):
            G.add_node(__node.node_id, **__node.model_dump())
        for __edge in self.edges.gen(relation=RelationIsA, conn=conn):
            G.add_edge(__edge.from_node_id, __edge.to_node_id, **__edge.model_dump())
        for __edge in self.edges.gen(relation=RelationHasA, conn=conn):
            G.add_edge(__edge.from_node_id, __edge.to_node_id, **__edge.model_dump())
        for __edge in self.edges.gen(relation=RelationRelatedTo, conn=conn):
            G.add_edge(__edge.from_node_id, __edge.to_node_id, **__edge.model_dump())
        for __edge in self.edges.gen(relation=RelationIsFrom, conn=conn):
            G.add_edge(__edge.from_node_id, __edge.to_node_id, **__edge.model_dump())
        return G

    def export(
        self,
        path: pathlib.Path | str,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        format: typing.Literal["node_link", "gexf"],
    ) -> pathlib.Path:
        from dvs.utils.dump_graph import dump_graph

        G = self.to_nx(conn=conn)
        return dump_graph(G, path, format=format)

    @functools.cached_property
    def nodes(self) -> "Nodes":
        from dvs.db.graph.nodes.api import Nodes

        return Nodes(self.dvs)

    @functools.cached_property
    def edges(self) -> "Edges":
        from dvs.db.graph.edges.api import Edges

        return Edges(self.dvs)

    @functools.cached_property
    def utils(self) -> "Utils":
        from dvs.db.graph.utils.api import Utils

        return Utils(self.dvs)

    @functools.cached_property
    def algorithm(self) -> "Algorithm":
        from dvs.db.graph.algorithm.api import Algorithm

        return Algorithm(self.dvs)

    async def search_vector_expansion(
        self,
        query: str,
        top_k: int = 3,
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
    ) -> list[GraphRAGResult]:
        """
        Simplified Strategy 1: Vector → Entity Discovery → Scoring → Results

        Algorithm Flow:
        Step 1: Vector search to find seed documents
        Step 2: Entity discovery & multi-relation graph expansion (CORE)
        Step 3: Scoring algorithm combining vector + graph relevance
        Final Step: Rank and format results

        Core concept: Start with vector search, expand through graph relationships
        (is_a, has_a, related_to), and score using weighted combination of
        vector similarity and entity overlap relevance.

        Pros: Clean separation of concerns, focused on core graph expansion logic.
        Cons: Simplified scoring without hub suppression or complex weighting.
        """

        conn = conn or self.dvs.new_connection()

        # Step 1: Vector search to find seed documents
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

        # Step 2: Entity Discovery & Graph Expansion (CORE CONCEPT)
        # Find entities connected to seed documents and expand through graph relations
        related_entities: set[str] = set()
        if document_ids:
            related_entities = (
                self.dvs.db.graph.utils.collect_entities_via_is_from_for_documents(
                    document_ids, limit_per_doc=10, conn=conn, verbose=False
                )
            )

        # Expand entities through multiple relation types for comprehensive coverage
        expanded_entity_ids = (
            self.dvs.db.graph.utils.expand_entities_with_multiple_relations(
                related_entities,
                is_a_max_hops=is_a_max_hops,
                is_a_limit_per_hop=is_a_limit_per_hop,
                has_a_enabled=has_a_enabled,
                has_a_limit_per_entity=has_a_limit_per_entity,
                related_to_enabled=related_to_enabled,
                related_to_limit_per_entity=related_to_limit_per_entity,
                suppress_hubs=False,  # Simplified version doesn't use hub suppression
                entity_expansion_cap=entity_expansion_cap,
                conn=conn,
                verbose=False,
            )
        )

        # Step 3: Scoring Algorithm - Combine vector and graph relevance
        # Pre-cache all document nodes to avoid repeated database calls
        doc_label_nodes: dict[str, "NodeType"] = {}
        all_doc_ids = set(document_ids)  # Original seed documents

        # Add current document IDs to cache
        for doc_node in self.dvs.db.graph.nodes.retrieve_by_labels(
            list(all_doc_ids), conn=conn, verbose=False
        ):
            if doc_node.node_id not in doc_label_nodes:
                doc_label_nodes[doc_node.node_id] = doc_node

        final_results: list[tuple[Document, float, float, float]] = []
        for __point, doc, vec_score in initial_results:
            try:
                # Use expanded entities directly for graph relevance calculation
                graph_score: float = self.dvs.db.graph.utils.calculate_graph_relevance(
                    doc.document_id,
                    document_ids,
                    expanded_entity_ids,
                    doc_label_nodes=doc_label_nodes,
                    conn=conn,
                    verbose=False,
                )
                # Weighted combination (CORE CONCEPT: weighted scoring)
                combined_score: float = (
                    vector_weight * float(vec_score) + graph_weight * graph_score
                )
                final_results.append(
                    (doc, combined_score, float(vec_score), graph_score)
                )
            except Exception:
                final_results.append((doc, float(vec_score), float(vec_score), 0.0))

        # Final Step: Rank and Format Results
        final_results.sort(key=lambda x: x[1], reverse=True)
        top = final_results[:top_k]

        out: list[GraphRAGResult] = []
        max_score: float = float(top[0][1]) if top else 1.0
        for idx, (doc, combined, vec_score, graph_score) in enumerate(top, start=1):
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=float(combined),
                    vector_score=float(vec_score),
                    graph_score=float(graph_score),
                    iterations=None,
                    rank=idx,
                    normalized_score=(
                        float(combined) / max_score if max_score > 0 else 0.0
                    ),
                    strategy="vector_expansion",
                )
            )

        if self.dvs.v(verbose):
            logger.info(
                f"🔍 [Simple Strategy 1] Query processed, found {len(out)} results"
            )

        return out

    async def search_graph_guided(
        self,
        query: str,
        top_k: int = 3,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
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

        conn = conn or self.dvs.new_connection()

        # Start timing
        start_time = time.perf_counter()

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

        pagerank_results = self.dvs.db.graph.algorithm.pagerank(
            relation=relation_type,  # type: ignore
            limit=top_k * 10,  # Get more candidates for filtering
            conn=conn,
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
                conn=conn,
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
                        outs = self.dvs.db.graph.utils.get_neighbors(
                            from_node_id_or_label=eid,
                            relation=RelationIsA,
                            limit=is_a_limit_per_hop,
                            conn=conn,
                            verbose=False,
                        )
                    except Exception:
                        outs = []
                    try:
                        ins = self.dvs.db.graph.utils.get_neighbors(
                            to_node_id_or_label=eid,
                            relation=RelationIsA,
                            limit=is_a_limit_per_hop,
                            conn=conn,
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
                    outs = self.dvs.db.graph.utils.get_neighbors(
                        from_node_id_or_label=eid,
                        relation=RelationHasA,
                        limit=has_a_limit_per_entity,
                        conn=conn,
                        verbose=False,
                    )
                except Exception:
                    outs = []
                try:
                    ins = self.dvs.db.graph.utils.get_neighbors(
                        to_node_id_or_label=eid,
                        relation=RelationHasA,
                        limit=has_a_limit_per_entity,
                        conn=conn,
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
                expanded_entities = self.dvs.db.graph.utils.suppress_hubs_by_pagerank(
                    expanded_entities,
                    hub_pagerank_top_percent,
                    conn=conn,
                    relation=RelationRelatedTo,
                    limit=5000,
                )
            except Exception:
                pass

        # Collect documents from expanded entities via is_from
        related_documents = set()
        for node_id in list(expanded_entities)[:1000]:
            try:
                neighbors = self.dvs.db.graph.utils.get_neighbors(
                    from_node_id_or_label=node_id,
                    relation=RelationIsFrom,
                    limit=5,
                    conn=conn,
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
                conn=conn,
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
        candidate_points: list["Point"] = (
            self.dvs.db.graph.utils.gather_points_for_documents(
                list(related_documents)[:50],
                per_doc_limit=10,
                with_embedding=True,
                conn=conn,
                verbose=self.dvs.v(verbose),
            )
        )

        if not candidate_points:
            logger.warning(
                "⚠️ No candidate points found, falling back to regular search"
            )
            base = await self.dvs.search(
                query,
                top_k=top_k,
                with_embedding=with_embedding,
                conn=conn,
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
        query_vector: list[float] = await asyncio.to_thread(
            self.dvs.utils.embed_text, query
        )

        scored_candidates: list[tuple[Point, Document, float, float]] = []
        for point in candidate_points:
            try:
                if not point.embedding:
                    raise ValueError("Point has no embedding")
                    continue

                point_vector = point.to_python()
                vector_similarity = cosine_similarity(query_vector, point_vector)

                # Find the document this point belongs to
                doc = self.dvs.db.documents.retrieve(
                    point.document_id, conn=conn, verbose=False
                )

                # Get graph importance score for this document's related entities
                graph_importance = 0.0
                for node_id, node_score in important_nodes:
                    try:
                        # Check if this document is related to the important entity
                        # Match by document label, not node_id
                        doc_node = self.dvs.db.graph.nodes.retrieve_by_label_or_raise(
                            doc.document_id, conn=conn, verbose=False
                        )
                        neighbors = self.dvs.db.graph.utils.get_neighbors(
                            from_node_id_or_label=node_id,
                            to_node_id_or_label=doc_node.node_id,
                            relation=RelationIsFrom,
                            limit=1,
                            conn=conn,
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

        # End timing and log performance
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(
            f"🔍 [Strategy 2: graph_guided] Query: '{query[:50]}...' | "
            f"Top-k: {top_k} | Duration: {duration_ms:.3f} ms | Results: {len(out)}"
        )

        return out

    async def search_hybrid_scoring(
        self,
        query: str,
        top_k: int = 3,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        vector_weight: float = 0.5,
        graph_importance_weight: float = 0.3,
        graph_distance_weight: float = 0.2,
        with_embedding: bool = False,
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

        # Start timing
        start_time = time.perf_counter()

        # cosine_similarity not needed in Strategy 3
        # Step 1: Initial vector search to get candidate points
        initial_results: list[tuple["Point", "Document", float]] = (
            await self.dvs.search(
                query=query,
                top_k=max(1, top_k * 3),
                with_embedding=with_embedding,
                conn=conn,
                verbose=self.dvs.v(verbose),
            )
        )

        if not initial_results:
            return []

        original_doc_ids: list[str] = [doc.document_id for _, doc, _ in initial_results]

        # Step 2: Prepare PageRank-based graph importance
        pagerank_results = self.dvs.db.graph.algorithm.pagerank(
            conn=conn,
            relation=RelationRelatedTo,  # Use general semantic connectivity
            limit=top_k * 50,
            verbose=self.dvs.v(verbose),
        )

        pagerank_map: dict[str, float] = {
            node_id: score for node_id, score in pagerank_results
        }
        max_pagerank: float = max(pagerank_map.values()) if pagerank_map else 1.0

        def get_graph_importance_for_document(document_id: str) -> float:
            return self.dvs.db.graph.utils.get_graph_importance_for_document(
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
            return self.dvs.db.graph.utils.get_graph_distance_score(
                document_id, original_doc_ids
            )

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

        # End timing and log performance
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(
            f"🔍 [Strategy 3: hybrid_scoring] Query: '{query[:50]}...' | "
            f"Top-k: {top_k} | Duration: {duration_ms:.3f} ms | Results: {len(out)}"
        )

        return out

    async def search_iterative_refinement(
        self,
        query: str,
        top_k: int = 3,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
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

        # Start timing
        start_time = time.perf_counter()

        if query_expander is None:
            raise ValueError("query_expander must be provided for LLM-based expansion.")

        # Step 0: Prepare baseline using original query
        base_vector: list[float] = await asyncio.to_thread(
            self.dvs.utils.embed_text, query
        )

        baseline_results = await VSS.vector_search(
            vector=base_vector,
            top_k=max(1, top_k),
            embedding_dimensions=self.dvs.db_manifest.embedding_dimensions,
            documents_table_name=dvs.DVS_DOCUMENTS_TABLE_NAME,
            points_table_name=dvs.DVS_POINTS_TABLE_NAME,
            conn=conn or self.dvs.new_connection(read_only=True),
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
            expanded_vectors = await asyncio.to_thread(
                self.dvs.utils.embed_texts, expanded_queries
            )

            # 2b) Graph-guided expansion (is_from 1-hop) to boost recall
            seed_doc_ids: list[str] = [doc.document_id for _, doc, _ in best_results]
            entity_ids: set[str] = (
                self.dvs.db.graph.utils.collect_entities_via_is_from_for_documents(
                    seed_doc_ids, limit_per_doc=300, conn=conn, verbose=False
                )
            )

            # Expand entities via is_a multi-hop, then has_a 1-hop
            expanded_entities_iter: set[str] = self.dvs.db.graph.utils.expand_entities(
                set(entity_ids),
                is_a_max_hops=is_a_max_hops,
                is_a_limit_per_hop=is_a_limit_per_hop,
                has_a_enabled=has_a_enabled,
                has_a_limit_per_entity=has_a_limit_per_entity,
                suppress_hubs=suppress_hubs,
                hub_pagerank_top_percent=hub_pagerank_top_percent,
                cap_total=None,
                conn=conn,
                verbose=False,
            )

            # Expand to new documents (caps: 8 per entity, 150 total)
            graph_doc_ids: set[str] = self.dvs.db.graph.utils.collect_docs_via_is_from(
                expanded_entities_iter,
                per_entity_limit=8,
                cap_entities=150,
                conn=conn,
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
            pts = self.dvs.db.graph.utils.gather_points_for_documents(
                list(new_graph_docs)[:150],
                per_doc_limit=3,
                with_embedding=True,
                conn=conn,
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
                conn=conn or self.dvs.new_connection(read_only=True),
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

        # End timing and log performance
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(
            f"🔍 [Strategy 4: iterative_refinement] Query: '{query[:50]}...' | "
            f"Top-k: {top_k} | Duration: {duration_ms:.3f} ms | "
            f"Results: {len(output)} | Iterations: {iterations}"
        )

        return output

    async def search_context_aware(
        self,
        query: str,
        top_k: int = 3,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        context_similarity_threshold: float = 0.7,
        max_expansion_steps: int = 2,
        with_embedding: bool = False,
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

        # Start timing
        start_time = time.perf_counter()

        conn = conn or self.dvs.new_connection()

        # 0) Baseline vector search to get seed context
        baseline_results: list[tuple[Point, Document, float]] = await self.dvs.search(
            query=query,
            top_k=max(1, top_k),
            with_embedding=True,  # need embeddings to compute centroids
            conn=conn,
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
                        conn=conn,
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
        query_vector: list[float] = await asyncio.to_thread(
            self.dvs.utils.embed_text, query
        )

        # 2) Single-step graph expansion via entities (is_from)
        #    Optionally respect max_expansion_steps>0; here we perform one step.
        if max_expansion_steps <= 0:
            max_expansion_steps = 1

        entity_ids: set[str] = set()
        for doc_node in self.dvs.db.graph.nodes.retrieve_by_labels(
            seed_doc_ids, conn=conn, verbose=False
        ):
            neighbors = self.dvs.db.graph.utils.get_neighbors(
                to_node_id_or_label=doc_node.node_id,
                relation=RelationIsFrom,
                limit=200,
                conn=conn,
                verbose=False,
            )
            for from_node, _edge, to_node in neighbors:
                if from_node.kind == "entity":
                    entity_ids.add(from_node.node_id)
                if to_node.kind == "entity":
                    entity_ids.add(to_node.node_id)

        if self.dvs.v(verbose):
            logger.debug(f"[S5] collected_entities={len(entity_ids)}")

        # 2b) Expand via is_a/has_a, then hub suppression via related_to PR
        expanded_entities5: set[str] = set(entity_ids)
        expanded_entities5 = self.dvs.db.graph.utils.expand_is_a_bfs(
            expanded_entities5,
            is_a_max_hops,
            is_a_limit_per_hop,
            cap_total=None,
            conn=conn,
            verbose=False,
        )
        if has_a_enabled and expanded_entities5:
            expanded_entities5 |= self.dvs.db.graph.utils.expand_has_a_one_hop(
                expanded_entities5,
                has_a_limit_per_entity,
                cap_total=None,
                conn=conn,
                verbose=False,
            )

        if suppress_hubs and expanded_entities5:
            before_hub5: int = len(expanded_entities5)
            expanded_entities5 = self.dvs.db.graph.utils.suppress_hubs_by_pagerank(
                expanded_entities5,
                hub_pagerank_top_percent,
                conn=conn,
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
        candidate_doc_ids: set[str] = self.dvs.db.graph.utils.collect_docs_via_is_from(
            expanded_entities5,
            per_entity_limit=8,
            cap_entities=300,
            conn=conn,
            verbose=False,
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

        # Apply a hard cap to candidate docs to control latency
        MAX_CANDIDATES: int = 200
        if len(candidate_doc_ids) > MAX_CANDIDATES:
            candidate_doc_ids = set(list(candidate_doc_ids)[:MAX_CANDIDATES])

        # 4) Score candidates by query similarity and context similarity
        alpha: float = 0.6  # weight for query-vs-candidate vector similarity
        beta: float = 0.4  # weight for context-vs-candidate similarity

        scored: list[tuple[Document, float, float, float]] = []
        log_counter: int = 0
        for doc_id in list(candidate_doc_ids)[:500]:
            try:
                # fewer points per doc for centroid to reduce DB calls
                cand_centroid: list[float] = (
                    self.dvs.db.graph.utils.centroid_for_document(
                        doc_id, conn=conn, limit_points=3
                    )
                )
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
                doc = self.dvs.db.documents.retrieve(doc_id, conn=conn, verbose=False)
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

        # End timing and log performance
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(
            f"🔍 [Strategy 5: context_aware] Query: '{query[:50]}...' | "
            f"Top-k: {top_k} | Duration: {duration_ms:.3f} ms | Results: {len(out)}"
        )

        return out
