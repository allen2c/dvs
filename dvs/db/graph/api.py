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
from dvs.types.document import Document
from dvs.types.edge import (
    RelationHasA,
    RelationIsA,
    RelationIsFrom,
    RelationRelatedTo,
)
from dvs.types.entity import Entity
from dvs.types.graphrag_result import GraphRAGResult
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

        # Use the shared vector expansion logic from utils
        enhanced_results = (
            await (
                self.dvs.db.graph.utils.perform_vector_expansion_search(
                    dvs=self.dvs,
                    query=query,
                    top_k=top_k,
                    conn=conn,
                    vector_weight=vector_weight,
                    graph_weight=graph_weight,
                    is_a_max_hops=is_a_max_hops,
                    is_a_limit_per_hop=is_a_limit_per_hop,
                    has_a_enabled=has_a_enabled,
                    has_a_limit_per_entity=has_a_limit_per_entity,
                    related_to_enabled=related_to_enabled,
                    related_to_limit_per_entity=related_to_limit_per_entity,
                    entity_expansion_cap=entity_expansion_cap,
                    verbose=self.dvs.v(verbose),
                )
            )
        )

        if not enhanced_results:
            return []

        # Convert to format expected by final processing
        final_results: list[tuple[Document, float, float, float]] = []
        for point, doc, combined_score in enhanced_results:
            # For Strategy 1, we need to separate vector and graph scores
            # We'll use the combined score as vector score and set graph score to 0
            # since the private method already combines them
            final_results.append(
                (
                    doc,
                    combined_score,
                    combined_score,
                    0.0,  # vector_score = combined, graph_score = 0
                )
            )

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
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """
        Simplified Strategy 2: PageRank-guided vector search.

        Core algorithm flow:
        Step 1: PageRank to find important nodes (CORE)
        Step 2: Entity expansion through graph relations
        Step 3: Collect documents and prepare candidates
        Step 4: Calculate combined scores and return results

        Core concept: Use graph importance (PageRank) to guide vector search direction
        instead of searching all documents blindly.
        """

        from dvs.utils.cosine_similarity import cosine_similarity

        conn = conn or self.dvs.new_connection()

        # Step 1: Find important nodes using PageRank (CORE CONCEPT)
        pagerank_results = self.dvs.db.graph.algorithm.pagerank(
            relation=RelationRelatedTo,
            limit=top_k * 5,  # Simplified: fewer candidates
            conn=conn,
            verbose=self.dvs.v(verbose),
        )

        if not pagerank_results:
            logger.warning("⚠️ No PageRank results found, returning empty results")
            return []

        # Step 2: Entity Discovery & Graph Expansion
        important_nodes = [(node_id, score) for node_id, score in pagerank_results]
        seed_entities: set[str] = {nid for (nid, _score) in important_nodes}

        # Use existing utils method for entity expansion (simplified)
        expanded_entities = self.dvs.db.graph.utils.expand_entities(
            seed_entities,
            is_a_max_hops=2,  # Fixed 2 hops for is_a
            is_a_limit_per_hop=10,  # Simplified limit
            has_a_enabled=True,
            has_a_limit_per_entity=5,  # Simplified limit
            suppress_hubs=False,  # Keep simple, no hub suppression
            cap_total=300,  # Reasonable cap to prevent explosion
            conn=conn,
            verbose=self.dvs.v(verbose),
        )

        # Step 3: Collect documents and prepare candidates
        related_documents = self.dvs.db.graph.utils.collect_docs_via_is_from(
            expanded_entities,
            per_entity_limit=3,
            cap_entities=200,
            conn=conn,
            verbose=False,
        )

        if not related_documents:
            logger.warning("⚠️ No related documents found, returning empty results")
            return []

        candidate_points = self.dvs.db.graph.utils.gather_points_for_documents(
            list(related_documents)[:30],
            per_doc_limit=5,
            with_embedding=True,
            conn=conn,
            verbose=self.dvs.v(verbose),
        )

        if not candidate_points:
            logger.warning("⚠️ No candidate points found, returning empty results")
            return []

        # Step 4: Calculate combined scores
        query_vector = await asyncio.to_thread(self.dvs.utils.embed_text, query)

        scored_candidates = []
        for point in candidate_points:
            try:
                if not point.embedding:
                    continue

                point_vector = point.to_python()
                vector_similarity = cosine_similarity(query_vector, point_vector)

                doc = self.dvs.db.documents.retrieve(
                    point.document_id, conn=conn, verbose=False
                )

                # Graph importance from PageRank scores
                graph_importance = 0.0
                for node_id, node_score in important_nodes:
                    try:
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

                # Weighted combination (CORE CONCEPT)
                combined_score = (
                    vector_weight * vector_similarity + graph_weight * graph_importance
                )

                scored_candidates.append(
                    (point, doc, combined_score, vector_similarity)
                )

            except Exception:
                continue

        # Final Step: Rank and format results
        scored_candidates.sort(key=lambda x: x[2], reverse=True)
        top = scored_candidates[:top_k]

        out = []
        max_score = float(top[0][2]) if top else 1.0
        for idx, (_p, doc, combined, vsc) in enumerate(top, start=1):
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=float(combined),
                    vector_score=float(vsc),
                    graph_score=0.0,  # Simplified: no graph score tracking
                    iterations=None,
                    rank=idx,
                    normalized_score=(
                        float(combined) / max_score if max_score > 0 else 0.0
                    ),
                    strategy="graph_guided",
                )
            )

        if self.dvs.v(verbose):
            logger.info(
                f"🔍 [Simplified Strategy 2] Query processed, found {len(out)} results"
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
        entity_expansion_cap: int = 200,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """
        Strategy 3: Simplified Iterative Refinement

        Core concept: Start with baseline search, then iteratively improve by:
        1. Using LLM to expand queries
        2. Using graph expansion to find related documents
        3. Combining expanded information for better search
        4. Stop when improvement becomes minimal

        Reuses existing utilities from graph.utils and search strategies.
        """
        # Start timing
        start_time = time.perf_counter()

        if query_expander is None:
            raise ValueError("query_expander must be provided for LLM-based expansion.")

        conn = conn or self.dvs.new_connection()

        # Step 1: Baseline search using enhanced vector expansion
        baseline_results = (
            await (
                self.dvs.db.graph.utils.perform_vector_expansion_search(
                    dvs=self.dvs,
                    query=query,
                    top_k=top_k,
                    conn=conn,
                    vector_weight=0.7,  # Use default weights for baseline
                    graph_weight=0.3,
                    is_a_max_hops=is_a_max_hops,
                    is_a_limit_per_hop=is_a_limit_per_hop,
                    has_a_enabled=has_a_enabled,
                    has_a_limit_per_entity=has_a_limit_per_entity,
                    related_to_enabled=True,  # Enable for baseline
                    related_to_limit_per_entity=1,
                    entity_expansion_cap=entity_expansion_cap,
                    verbose=self.dvs.v(verbose),
                )
            )
        )

        if not baseline_results:
            return []

        best_results = baseline_results
        best_top1: float = float(baseline_results[0][2])
        iterations: int = 0

        if self.dvs.v(verbose):
            logger.info(f"[Iterative] Baseline top1={best_top1:.3f}")

        # Iterative refinement loop
        while iterations < max_iterations:
            # Step 2: LLM query expansion
            expanded_queries: list[str] = [
                q
                for q in (await query_expander(query))[:expansions_per_iter]
                if isinstance(q, str) and q.strip()
            ]

            if not expanded_queries:
                if self.dvs.v(verbose):
                    logger.info("[Iterative] No expanded queries, stopping")
                break

            # Step 3: Graph-guided expansion (reuse existing utilities)
            seed_doc_ids = [doc.document_id for _, doc, _ in best_results]

            # Get entities from current best documents
            entity_ids = (
                self.dvs.db.graph.utils.collect_entities_via_is_from_for_documents(
                    seed_doc_ids,
                    limit_per_doc=50,
                    conn=conn,
                    verbose=False,
                )
            )

            # Expand entities using existing utility
            expanded_entities = self.dvs.db.graph.utils.expand_entities(
                entity_ids,
                is_a_max_hops=is_a_max_hops,
                is_a_limit_per_hop=is_a_limit_per_hop,
                has_a_enabled=has_a_enabled,
                has_a_limit_per_entity=has_a_limit_per_entity,
                suppress_hubs=suppress_hubs,
                hub_pagerank_top_percent=hub_pagerank_top_percent,
                cap_total=200,
                conn=conn,
                verbose=False,
            )

            # Get new documents from expanded entities
            graph_doc_ids = self.dvs.db.graph.utils.collect_docs_via_is_from(
                expanded_entities,
                per_entity_limit=5,
                cap_entities=100,
                conn=conn,
                verbose=False,
            )

            new_docs = graph_doc_ids.difference(set(seed_doc_ids))
            if not new_docs:
                if self.dvs.v(verbose):
                    logger.info("[Iterative] No new documents from graph, stopping")
                break

            # Step 4: Combine information for improved search
            # Use expanded queries with enhanced vector expansion
            expanded_results: list[tuple[Point, Document, float]] = []
            for exp_query in expanded_queries:
                results = await self.dvs.db.graph.utils.perform_vector_expansion_search(
                    dvs=self.dvs,
                    query=exp_query,
                    top_k=top_k,
                    conn=conn,
                    vector_weight=0.7,  # Use same weights as baseline
                    graph_weight=0.3,
                    is_a_max_hops=is_a_max_hops,
                    is_a_limit_per_hop=is_a_limit_per_hop,
                    has_a_enabled=has_a_enabled,
                    has_a_limit_per_entity=has_a_limit_per_entity,
                    related_to_enabled=True,
                    related_to_limit_per_entity=1,
                    entity_expansion_cap=entity_expansion_cap,
                    verbose=False,
                )
                expanded_results.extend(results)

            # Combine and deduplicate results
            all_candidates = list(set(expanded_results + baseline_results))
            if len(all_candidates) > top_k * 2:
                all_candidates = all_candidates[: top_k * 2]

            # Step 5: Check improvement
            if all_candidates:
                improved_top1 = float(all_candidates[0][2])
                improvement = improved_top1 - best_top1

                if improvement <= refinement_threshold:
                    if self.dvs.v(verbose):
                        logger.info(
                            f"[Iterative] Improvement {improvement:.3f} < threshold, "
                            "stopping",
                        )
                    break

                # Accept improvement
                best_results = all_candidates
                best_top1 = improved_top1
                iterations += 1

                if self.dvs.v(verbose):
                    logger.info(
                        f"[Iterative] Iter {iterations}: improved to {best_top1:.3f}",
                    )
            else:
                break

        # Step 6: Format final results
        final_results = best_results[:top_k]
        output = [
            GraphRAGResult(
                document=doc,
                score=float(score),
                vector_score=float(score),
                graph_score=None,
                iterations=iterations,
                rank=i + 1,
                normalized_score=(
                    float(score) / float(final_results[0][2]) if final_results else 0.0
                ),
                strategy="iterative_refinement",
            )
            for i, (_point, doc, score) in enumerate(final_results)
        ]

        # End timing and log performance
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000
        logger.info(
            f"🔍 [Strategy 3: iterative_refinement] Query: '{query[:50]}...' | "
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
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """
        Simplified Strategy 4: Context-Aware Search

        Core concept: Use context centroid to prevent semantic drift.
        1. Baseline vector search → get seed documents
        2. Build context centroid from seed embeddings
        3. Simple entity expansion via is_from
        4. Score candidates with dual similarity (query + context)
        5. Filter by context threshold to maintain relevance

        Simplified: Removed complex hub suppression, multi-hop expansion,
        and excessive logging. Focus on core context-aware scoring.
        """
        from dvs.utils.cosine_similarity import cosine_similarity
        from dvs.utils.graph_ops import mean_vector

        start_time = time.perf_counter()
        conn = conn or self.dvs.new_connection()

        # Step 1: Baseline vector search to get seed context
        baseline_results: list[tuple[Point, Document, float]] = await self.dvs.search(
            query=query,
            top_k=max(3, top_k),  # Get more seeds for better centroid
            with_embedding=True,
            conn=conn,
            verbose=self.dvs.v(verbose),
        )

        if not baseline_results:
            return []

        # Step 2: Build context centroid from seed embeddings
        seed_vectors: list[list[float]] = []
        seed_doc_ids: list[str] = []

        for pt, doc, _ in baseline_results:
            if pt.embedding:
                seed_vectors.append(pt.to_python())
            seed_doc_ids.append(doc.document_id)

        # Fallback to fetch embeddings if none found
        if not seed_vectors:
            for doc in baseline_results:
                pts = self.dvs.db.points.gen(
                    document_id=doc[1].document_id,
                    limit=3,
                    with_embedding=True,
                    conn=conn,
                    verbose=False,
                )
                for p in pts:
                    if p.embedding:
                        seed_vectors.append(p.to_python())

        # If still no embeddings, return baseline results
        if not seed_vectors:
            max_score = float(baseline_results[0][2]) if baseline_results else 1.0
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=0.0,
                    iterations=0,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score if max_score > 0 else 0.0),
                    strategy="context_aware",
                )
                for i, (_p, doc, sc) in enumerate(baseline_results[:top_k])
            ]

        context_centroid: list[float] = mean_vector(seed_vectors)
        query_vector: list[float] = await asyncio.to_thread(
            self.dvs.utils.embed_text, query
        )

        # Step 3: Simple entity expansion via is_from
        entity_ids: set[str] = set()
        for doc_node in self.dvs.db.graph.nodes.retrieve_by_labels(
            seed_doc_ids, conn=conn, verbose=False
        ):
            neighbors = self.dvs.db.graph.utils.get_neighbors(
                to_node_id_or_label=doc_node.node_id,
                relation=RelationIsFrom,
                limit=50,  # Simplified limit
                conn=conn,
                verbose=False,
            )
            for from_node, _edge, to_node in neighbors:
                if getattr(from_node, "kind", None) == "entity":
                    entity_ids.add(from_node.node_id)
                if getattr(to_node, "kind", None) == "entity":
                    entity_ids.add(to_node.node_id)

        # Step 4: Collect candidate documents (simplified - no complex expansion)
        candidate_doc_ids: set[str] = self.dvs.db.graph.utils.collect_docs_via_is_from(
            entity_ids,
            per_entity_limit=5,  # Simplified limit
            cap_entities=100,  # Simplified cap
            conn=conn,
            verbose=False,
        )

        # Remove seed documents from candidates
        candidate_doc_ids.difference_update(set(seed_doc_ids))

        if not candidate_doc_ids:
            # Return baseline if no expansion possible
            max_score = float(baseline_results[0][2]) if baseline_results else 1.0
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=0.0,
                    iterations=0,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score if max_score > 0 else 0.0),
                    strategy="context_aware",
                )
                for i, (_p, doc, sc) in enumerate(baseline_results[:top_k])
            ]

        # Step 5: Dual similarity scoring with context filtering
        scored: list[tuple[Document, float, float, float]] = []
        alpha: float = 0.6  # Query similarity weight
        beta: float = 0.4  # Context similarity weight

        for doc_id in list(candidate_doc_ids)[:200]:  # Cap candidates
            try:
                cand_centroid: list[float] = (
                    self.dvs.db.graph.utils.centroid_for_document(
                        doc_id, conn=conn, limit_points=3
                    )
                )

                if not cand_centroid:
                    continue

                vec_sim: float = float(cosine_similarity(query_vector, cand_centroid))
                ctx_sim: float = float(
                    cosine_similarity(context_centroid, cand_centroid)
                )

                # Context threshold filter (CORE CONCEPT)
                if ctx_sim < context_similarity_threshold:
                    continue

                combined_score: float = alpha * vec_sim + beta * ctx_sim
                doc = self.dvs.db.documents.retrieve(doc_id, conn=conn, verbose=False)
                scored.append((doc, combined_score, vec_sim, ctx_sim))

            except Exception:
                continue

        # Fallback to baseline if no candidates pass threshold
        if not scored:
            max_score = float(baseline_results[0][2]) if baseline_results else 1.0
            return [
                GraphRAGResult(
                    document=doc,
                    score=float(sc),
                    vector_score=float(sc),
                    graph_score=0.0,
                    iterations=0,
                    rank=i + 1,
                    normalized_score=(float(sc) / max_score if max_score > 0 else 0.0),
                    strategy="context_aware",
                )
                for i, (_p, doc, sc) in enumerate(baseline_results[:top_k])
            ]

        # Step 6: Rank and format results
        scored.sort(key=lambda x: x[1], reverse=True)
        top = scored[:top_k]
        max_score: float = float(top[0][1]) if top else 1.0

        out: list[GraphRAGResult] = []
        for idx, (doc, combined, vec_sim, ctx_sim) in enumerate(top, start=1):
            out.append(
                GraphRAGResult(
                    document=doc,
                    score=float(combined),
                    vector_score=float(vec_sim),
                    graph_score=float(ctx_sim),
                    iterations=1,
                    rank=idx,
                    normalized_score=(
                        float(combined) / max_score if max_score > 0 else 0.0
                    ),
                    strategy="context_aware",
                )
            )

        # Simple performance logging
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            f"🔍 [Context-Aware] Query processed | Top-k: {top_k} | "
            f"Duration: {duration_ms:.3f} ms | Results: {len(out)}"
        )

        return out

    async def search_default(
        self,
        query: str,
        top_k: int = 3,
        *,
        conn: duckdb.DuckDBPyConnection | None = None,
        with_embedding: bool = True,
        strategy: typing.Literal[
            "default",
            "vector_expansion",
            "graph_guided",
            "iterative_refinement",
            "context_aware",
        ] = "default",
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """
        Default fallback search strategy using basic vector search.

        This method provides a standardized fallback when graph-based strategies
        cannot find sufficient results.
        """
        conn = conn or self.dvs.new_connection()

        # Basic vector search
        base = await self.dvs.search(
            query=query,
            top_k=top_k,
            with_embedding=with_embedding,
            conn=conn,
            verbose=self.dvs.v(verbose),
        )

        max_score = float(base[0][2]) if base else 1.0
        out = [
            GraphRAGResult(
                document=doc,
                score=float(sc),
                vector_score=float(sc),
                graph_score=0.0,
                iterations=None,
                rank=i + 1,
                normalized_score=(float(sc) / max_score) if max_score > 0 else 0.0,
                strategy=strategy,
            )
            for i, (_p, doc, sc) in enumerate(base)
        ]

        if self.dvs.v(verbose):
            logger.info(
                f"🔍 [Default Fallback] Query processed, found {len(out)} results"
            )

        return out
