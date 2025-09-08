import abc
import functools
import logging
import pathlib
import typing

import duckdb
import openai
import openai_embeddings_model as oai_emb_model
from str_or_none import str_or_none
from tqdm import tqdm

import dvs
import dvs.utils.vss as VSS
from dvs.config import Settings
from dvs.types.document import Document
from dvs.types.encoding_type import EncodingType
from dvs.types.graphrag_result import GraphRAGResult
from dvs.types.point import Point
from dvs.types.search_request import SearchRequest
from dvs.utils.chunk import chunks

if typing.TYPE_CHECKING:
    from dvs.db.api import DB
    from dvs.tokens import Tokens
    from dvs.types.manifest import Manifest as ManifestType


logger = logging.getLogger(__name__)


class DVSMixin(abc.ABC):
    db: "DB"
    model: oai_emb_model.OpenAIEmbeddingsModel
    model_settings: oai_emb_model.ModelSettings
    verbose: bool

    def _ensure_dvs_settings(
        self, settings: typing.Union[pathlib.Path, str] | Settings
    ) -> Settings:
        """Ensure DVS settings are properly configured and validated."""
        if isinstance(settings, Settings):
            pass
        else:
            settings = Settings(DUCKDB_PATH=str(settings))

        if settings.DUCKDB_PATH is None:
            raise ValueError("DUCKDB_PATH is not set")

        return settings

    def _ensure_model(
        self, model: oai_emb_model.OpenAIEmbeddingsModel | str
    ) -> oai_emb_model.OpenAIEmbeddingsModel:
        """Ensure OpenAI embeddings model is properly initialized."""
        if isinstance(model, oai_emb_model.OpenAIEmbeddingsModel):
            return model
        else:
            return oai_emb_model.OpenAIEmbeddingsModel(
                model, openai.OpenAI(), cache=oai_emb_model.get_default_cache()
            )

    def _ensure_manifest(
        self,
        model: oai_emb_model.OpenAIEmbeddingsModel,
        model_settings: oai_emb_model.ModelSettings,
        verbose: bool,
    ) -> "ManifestType":
        """
        Ensure database manifest is consistent with model and model settings.
        Creates manifest if missing or validates existing one against current model.
        Sets dimensions in model_settings if None and returns the manifest.
        """  # noqa: E501
        from dvs.types.manifest import Manifest as ManifestType

        # Ensure the manifest table exists
        if dvs.DVS_MANIFEST_TABLE_NAME not in self.db.show_table_names():
            logger.debug("Manifest table does not exist, creating it")
            self.db.manifest.touch(verbose=verbose)

        might_manifest = self.db.manifest.receive(verbose=verbose)

        # If the manifest table exists but is empty, create a new manifest
        if might_manifest is None:
            logger.debug("Manifest table is empty, creating a new manifest")
            if model_settings.dimensions is None:
                raise ValueError(
                    "Could not infer the embedding dimensions, "
                    + "please provide the model settings."
                )

            self.db_manifest = self.db.manifest.create(
                ManifestType(
                    embedding_model=self.model.model,
                    embedding_dimensions=model_settings.dimensions,
                ),
                verbose=verbose,
            )

        # If the manifest table exists and is not empty, use the existing manifest
        else:
            logger.debug("Manifest table exists, using the existing manifest")
            self.db_manifest = might_manifest

            if self.db_manifest.embedding_model != model.model:
                raise ValueError(
                    "The indicated embedding model is not the same as "
                    + "the one in the manifest of the database"
                )
            if model_settings.dimensions is not None:
                if self.db_manifest.embedding_dimensions != model_settings.dimensions:
                    raise ValueError(
                        "The indicated embedding dimensions are not the same as "
                        + "the one in the manifest of the database"
                    )
            else:
                model_settings.dimensions = self.db_manifest.embedding_dimensions

        return self.db_manifest


class DVS(DVSMixin):
    def __init__(
        self,
        settings: typing.Union[pathlib.Path, str] | Settings,
        *,
        model_settings: oai_emb_model.ModelSettings | None = None,
        model: oai_emb_model.OpenAIEmbeddingsModel | str,
        enable_graph: bool = False,
        verbose: bool | None = None,
    ):
        self.settings = self._ensure_dvs_settings(settings)
        self.verbose = verbose or False
        self.model = self._ensure_model(model)
        self.model_settings = model_settings or oai_emb_model.ModelSettings()
        self.enable_graph = enable_graph

        # Init database resources
        self.db_manifest = self._ensure_manifest(
            self.model, self.model_settings, verbose=self.verbose
        )

        self.db.touch(enable_graph=self.enable_graph, verbose=self.verbose)

    @property
    def duckdb_path(self) -> pathlib.Path:
        """Get the path to the DuckDB database file."""
        return self.settings.duckdb_path

    def new_connection(self, read_only: bool = False) -> duckdb.DuckDBPyConnection:
        """Always use a new duckdb connection."""
        conn = duckdb.connect(self.duckdb_path, read_only=read_only)
        return conn

    def add(
        self,
        documents: typing.Union[
            Document,
            typing.Iterable[Document],
            str,
            typing.Iterable[str],
            typing.Iterable[typing.Union[Document, str]],
        ],
        *,
        batch_size: int = 100,
        ignore_same_content: bool = True,
        lines_per_chunk: int = 80,
        tokens_per_chunk: int = 1200,
        verbose: bool | None = None,
    ) -> typing.Dict:
        """Add docs: chunk, embed, and store in DuckDB; return creation stats."""

        # Validate documents
        docs: list["Document"] = Document.from_contents(documents)
        ignored_docs_indexes: list[int] = []
        creating_points_count: int = 0

        # Chunk documents
        chunked_docs = [
            chunked_doc
            for doc in tqdm(
                docs,
                total=len(docs),
                disable=not self.v(verbose),
                desc="Chunking documents",
            )
            for chunked_doc in doc.to_chunked_documents(
                lines_per_chunk=lines_per_chunk,
                tokens_per_chunk=tokens_per_chunk,
                encoding=self.tokens.enc,
            )
        ]
        logger.debug(f"Chunked into {len(chunked_docs)} documents")

        # Collect documents
        for idx, doc in tqdm(
            enumerate(chunked_docs),
            total=len(chunked_docs),
            disable=not verbose,
            desc="Checking for duplicate documents",
        ):
            if ignore_same_content:
                if self.db.documents.content_exists(doc.content_md5, verbose=False):
                    logger.warning(
                        f"Document {repr(doc.name)[:12]} with content_md5 "
                        + f"'{doc.content_md5}' already exists, skipping creation"
                    )
                    ignored_docs_indexes.append(idx)
                    continue
        creating_docs = [
            doc
            for idx, doc in enumerate(chunked_docs)
            if idx not in ignored_docs_indexes
        ]

        # Create documents into the database
        self.db.documents.bulk_create(creating_docs, verbose=verbose)

        # Create embeddings (assign embeddings to points in place)
        for batch_docs in chunks(creating_docs, batch_size=batch_size):
            _pts_with_contents = [
                doc.to_point_with_content(with_embeddings=False) for doc in batch_docs
            ]
            _embeddings_resp = self.model.get_embeddings(
                input=[c for _, c in _pts_with_contents],
                model_settings=self.model_settings,
            )
            for (pt, _), embedding in zip(_pts_with_contents, _embeddings_resp.output):
                pt.embedding = embedding
                creating_points_count += 1

            self.db.points.bulk_create(
                [pt for pt, _ in _pts_with_contents],
                verbose=verbose,
            )

        return {
            "success": True,
            "created_documents": len(creating_docs),
            "ignored_documents": len(ignored_docs_indexes),
            "created_points": creating_points_count,
            "error": None,
        }

    def remove(
        self,
        doc_ids: typing.Union[str, typing.Iterable[str]],
        *,
        verbose: bool | None = None,
    ) -> None:
        """Remove documents and their points by IDs; irreversible if executed."""
        doc_ids = [doc_ids] if isinstance(doc_ids, str) else list(doc_ids)

        self.db.points.remove_many(document_ids=doc_ids, verbose=self.v(verbose))
        for doc_id in doc_ids:
            self.db.documents.remove(doc_id, verbose=self.v(verbose))

        return None

    async def search(
        self,
        query: str,
        top_k: int = 3,
        *,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[tuple["Point", "Document", float]]:
        """
        Perform asynchronous vector similarity search using text query.
        Converts query to embedding via OpenAI API and searches DuckDB using cosine similarity.
        Returns list of tuples containing matched point, document, and relevance score.
        """  # noqa: E501

        sanitized_query = str_or_none(query)
        if sanitized_query is None:
            raise ValueError("Query cannot be empty")

        # Validate search request
        search_req = SearchRequest.model_validate(
            {"query": query, "top_k": top_k, "with_embedding": with_embedding}
        )
        vectors = await SearchRequest.to_vectors(
            [search_req],
            model=self.model,
            model_settings=self.model_settings,
        )
        vector = vectors[0]

        # Perform vector search
        results = await VSS.vector_search(
            vector=vector,
            top_k=search_req.top_k,
            embedding_dimensions=self.db_manifest.embedding_dimensions,
            documents_table_name=dvs.DVS_DOCUMENTS_TABLE_NAME,
            points_table_name=dvs.DVS_POINTS_TABLE_NAME,
            conn=self.new_connection(read_only=True),
            with_embedding=search_req.with_embedding,
            debug=self.v(verbose),
            console=self.settings.console,
        )

        return results

    async def graph_rag_search_vector_expansion(
        self,
        query: str,
        top_k: int = 3,
        *,
        graph_expansion_depth: int = 1,
        vector_weight: float = 0.7,
        graph_weight: float = 0.3,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 1: vector search → graph expand → re-score (vector+graph).
        Pros: simple; leverages entities near seed docs; explainable via edges.
        Cons: sensitive to initial recall; may drift and pull hubs; extra passes.
        Use when: you want quick recall boost from doc-seeded entities and moderate
        latency is acceptable; centrality metrics are unavailable.
        """

        # Step 1: Vector search - Find most relevant documents
        initial_results = await self.search(
            query=query,
            top_k=top_k * 2,  # Expand candidate set
            with_embedding=with_embedding,
            verbose=self.v(verbose),
        )

        if not initial_results:
            return []

        # Step 2: Find corresponding document nodes from document IDs
        document_ids = [doc.document_id for _, doc, _ in initial_results]
        document_nodes = []

        for doc_id in document_ids:
            try:
                # Try to find corresponding document node (label = document_id)
                doc_node = self.db.graph.nodes.retrieve_by_label(
                    doc_id, verbose=self.v(verbose)
                )
                document_nodes.append(doc_node)
            except Exception:
                # Skip if corresponding node not found
                continue

        # Step 3: Find related entity nodes through "is_from" relationship
        related_entities = set()
        for doc_node in document_nodes:
            try:
                # Find all entity nodes connected to this document
                from dvs.types.edge import RelationIsFrom

                neighbors = self.db.graph.get_neighbors(
                    to_node_id_or_label=doc_node.node_id,
                    relation=RelationIsFrom,
                    limit=10,
                    verbose=self.v(verbose),
                )

                # Collect entity node IDs
                for _, _, entity_node in neighbors:
                    if entity_node.kind == "entity":
                        related_entities.add(entity_node.node_id)
            except Exception:
                continue

        # Step 4: Graph expansion on these entity nodes to find more related documents
        expanded_document_ids = set(document_ids)

        if graph_expansion_depth > 0:
            for entity_id in list(related_entities)[:20]:  # Limit processing count
                try:
                    # Find all document nodes connected to this entity
                    from dvs.types.edge import RelationIsFrom

                    neighbors = self.db.graph.get_neighbors(
                        from_node_id_or_label=entity_id,
                        relation=RelationIsFrom,
                        limit=5,
                        verbose=self.v(verbose),
                    )

                    # Collect new document IDs
                    for _, _, doc_node in neighbors:
                        if doc_node.kind == "document":
                            # Use label as document_id
                            expanded_document_ids.add(doc_node.label)
                except Exception:
                    continue

        # Step 5: Re-vector search on expanded document collection
        expanded_candidates = []
        for doc_id in expanded_document_ids:
            try:
                # Get all points for the document
                points = self.db.points.gen(
                    document_id=doc_id, limit=10, verbose=self.v(verbose)
                )
                for point in points:
                    if point.embedding:
                        expanded_candidates.append(point)
            except Exception:
                continue

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
        query_vector = (
            await SearchRequest.to_vectors(
                [SearchRequest(query=query, top_k=1)],
                model=self.model,
                model_settings=self.model_settings,
            )
        )[0]

        scored_candidates = []
        for point in expanded_candidates:
            try:
                point_vector = point.to_python()
                # Calculate cosine similarity
                similarity = self._cosine_similarity(query_vector, point_vector)
                scored_candidates.append((point, similarity))
            except Exception:
                continue

        # Step 7: Calculate combined score (vector similarity + graph relevance)
        final_results: list[tuple[Point, Document, float, float, float]] = []
        for point, vector_score in scored_candidates:
            try:
                doc = self.db.documents.retrieve(
                    point.document_id, verbose=self.v(verbose)
                )

                # Calculate graph relevance: based on distance to original
                # query-related documents
                graph_score = self._calculate_graph_relevance(
                    point.document_id,
                    document_ids,
                    related_entities,
                    verbose=self.v(verbose),
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

    async def graph_rag_search_graph_guided(
        self,
        query: str,
        top_k: int = 3,
        *,
        relation_types: list[str] | None = None,
        centrality_threshold: float = 0.5,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 2: PageRank-guided; vector over salient nodes, combine scores.
        Pros: robust for ambiguous queries; targets globally important graph areas.
        Cons: depends on centrality quality; may miss niche items; algo overhead.
        Use when: graph is rich and centrality is meaningful; discovery-oriented
        retrieval benefits from salient-node guidance.
        """
        from dvs.types.edge import RelationIsFrom, RelationRelatedTo

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

        pagerank_results = self.db.graph.pagerank(
            relation=relation_type,  # type: ignore
            limit=top_k * 10,  # Get more candidates for filtering
            verbose=self.v(verbose),
        )

        if not pagerank_results:
            logger.warning(
                "⚠️ No PageRank results found, falling back to regular search"
            )
            base = await self.search(
                query,
                top_k=top_k,
                with_embedding=with_embedding,
                verbose=self.v(verbose),
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

        related_documents = set()
        for node_id, graph_score in important_nodes:
            try:
                # Find documents connected to this entity node
                # via "is_from" relationship
                neighbors = self.db.graph.get_neighbors(
                    from_node_id_or_label=node_id,
                    relation=RelationIsFrom,
                    limit=5,  # Limit documents per entity
                    verbose=self.v(verbose),
                )

                for _, _, doc_node in neighbors:
                    if doc_node.kind == "document":
                        # Use label as document_id
                        related_documents.add(doc_node.label)

            except Exception as e:
                logger.error(f"⚠️ Error getting neighbors for node {node_id}: {e}")
                continue

        if not related_documents:
            logger.warning(
                "⚠️ No related documents found, falling back to regular search"
            )
            base = await self.search(
                query,
                top_k=top_k,
                with_embedding=with_embedding,
                verbose=self.v(verbose),
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
        candidate_points: list["Point"] = []
        # Limit to avoid too many candidates
        for doc_id in list(related_documents)[:50]:
            try:
                points = self.db.points.gen(
                    document_id=doc_id,
                    limit=10,
                    with_embedding=True,
                    verbose=self.v(verbose),
                )
                candidate_points.extend(points)
            except Exception as e:
                logger.error(f"⚠️ Error getting points for document {doc_id}: {e}")
                continue

        if not candidate_points:
            logger.warning(
                "⚠️ No candidate points found, falling back to regular search"
            )
            base = await self.search(
                query,
                top_k=top_k,
                with_embedding=with_embedding,
                verbose=self.v(verbose),
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
        query_vector = (
            await SearchRequest.to_vectors(
                [SearchRequest(query=query, top_k=1)],
                model=self.model,
                model_settings=self.model_settings,
            )
        )[0]

        scored_candidates: list[tuple[Point, Document, float, float]] = []
        for point in candidate_points:
            try:
                if not point.embedding:
                    raise ValueError("Point has no embedding")
                    continue

                point_vector = point.to_python()
                vector_similarity = self._cosine_similarity(query_vector, point_vector)

                # Find the document this point belongs to
                doc = self.db.documents.retrieve(point.document_id, verbose=False)

                # Get graph importance score for this document's related entities
                graph_importance = 0.0
                for node_id, node_score in important_nodes:
                    try:
                        # Check if this document is related to the important entity
                        # Match by document label, not node_id
                        doc_node = self.db.graph.nodes.retrieve_by_label(
                            doc.document_id, verbose=False
                        )
                        neighbors = self.db.graph.get_neighbors(
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

    async def graph_rag_search_hybrid_scoring(
        self,
        query: str,
        top_k: int = 3,
        *,
        vector_weight: float = 0.5,
        graph_importance_weight: float = 0.3,
        graph_distance_weight: float = 0.2,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 3: weighted blend of vector, graph importance, and distance.
        Pros: balances precision/recall; uses structure+semantics; explainable parts.
        Cons: needs weight tuning; adds PR/shortest-path cost; double-count risk.
        Use when: you can calibrate weights offline and want stable cross-domain
        performance.
        """
        from dvs.types.edge import RelationIsFrom, RelationRelatedTo

        # Step 1: Initial vector search to get candidate points
        initial_results: list[tuple["Point", "Document", float]] = await self.search(
            query=query,
            top_k=max(1, top_k * 3),
            with_embedding=with_embedding,
            verbose=self.v(verbose),
        )

        if not initial_results:
            return []

        original_doc_ids: list[str] = [doc.document_id for _, doc, _ in initial_results]

        # Step 2: Prepare PageRank-based graph importance
        pagerank_results = self.db.graph.pagerank(
            relation=RelationRelatedTo,  # Use general semantic connectivity
            limit=top_k * 50,
            verbose=self.v(verbose),
        )

        pagerank_map: dict[str, float] = {
            node_id: score for node_id, score in pagerank_results
        }
        max_pagerank: float = max(pagerank_map.values()) if pagerank_map else 1.0

        def get_graph_importance_for_document(document_id: str) -> float:
            """Return normalized graph importance [0,1] for a document node."""
            # Prefer PageRank on the document node directly if available
            if document_id in pagerank_map and max_pagerank > 0:
                return pagerank_map[document_id] / max_pagerank

            # Otherwise, look at connected entity nodes via is_from and take max
            try:
                neighbors = self.db.graph.get_neighbors(
                    to_node_id_or_label=document_id,
                    relation=RelationIsFrom,
                    limit=15,
                    verbose=False,
                )
                best: float = 0.0
                for _, _, to_node in neighbors:
                    node_score = pagerank_map.get(to_node.node_id, 0.0)
                    if max_pagerank > 0:
                        best = max(best, node_score / max_pagerank)
                return best
            except Exception:
                return 0.0

        # Step 3: Compute distance-based score using shortest paths to originals
        # Reuse _calculate_graph_relevance which maps distance to [0,1]
        def get_graph_distance_score(document_id: str) -> float:
            """Return distance score [0,1] derived from shortest path to seeds."""
            return self._calculate_graph_relevance(
                document_id,
                original_doc_ids,
                related_entities=set(),
                verbose=False,
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
        return out

    async def graph_rag_search_iterative_refinement(
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
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 4: iterate expand→embed→search until improvement is small.
        Pros: escapes local minima; adapts via feedback; good for exploration.
        Cons: multi-round latency; expander quality critical; tuning convergence.
        Use when: batch/offline or higher-latency is fine; need high recall and
        adaptive retrieval.
        """
        from dvs.types.edge import RelationIsFrom, RelationRelatedTo

        if query_expander is None:
            raise ValueError("query_expander must be provided for LLM-based expansion.")

        # Step 0: Prepare baseline using original query
        base_vector: list[float] = (
            await SearchRequest.to_vectors(
                [SearchRequest(query=query, top_k=1, encoding=EncodingType.PLAINTEXT)],
                model=self.model,
                model_settings=self.model_settings,
            )
        )[0]

        baseline_results = await VSS.vector_search(
            vector=base_vector,
            top_k=max(1, top_k),
            embedding_dimensions=self.db_manifest.embedding_dimensions,
            documents_table_name=dvs.DVS_DOCUMENTS_TABLE_NAME,
            points_table_name=dvs.DVS_POINTS_TABLE_NAME,
            conn=self.new_connection(read_only=True),
            with_embedding=False,
            debug=self.v(verbose),
            console=self.settings.console,
        )

        if not baseline_results:
            return []

        best_results: list[tuple[Point, Document, float]] = baseline_results
        best_top1: float = float(baseline_results[0][2])
        current_vector: list[float] = list(base_vector)
        iterations: int = 0

        if self.v(verbose):
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
            expanded_vectors: list[list[float]] = await SearchRequest.to_vectors(
                [
                    SearchRequest(query=eq, top_k=1, encoding=EncodingType.PLAINTEXT)
                    for eq in expanded_queries
                ],
                model=self.model,
                model_settings=self.model_settings,
            )

            # 2b) Graph-guided expansion (is_from 1-hop) to boost recall
            seed_doc_ids: list[str] = [doc.document_id for _, doc, _ in best_results]
            entity_ids: set[str] = set()
            for doc_id in seed_doc_ids:
                try:
                    doc_node = self.db.graph.nodes.retrieve_by_label(
                        doc_id, verbose=False
                    )
                    # Preferred direction: entity -> document (to_node == doc)
                    neighbors = self.db.graph.get_neighbors(
                        to_node_id_or_label=doc_node.node_id,
                        relation=RelationIsFrom,
                        limit=300,
                        verbose=False,
                    )
                    for from_node, _edge, to_node in neighbors:
                        # Accept whichever side is the entity
                        if from_node.kind == "entity":
                            entity_ids.add(from_node.node_id)
                        if to_node.kind == "entity":
                            entity_ids.add(to_node.node_id)
                except Exception:
                    continue

            # Suppress hubs using PageRank threshold
            pr = self.db.graph.pagerank(
                relation=RelationRelatedTo, limit=5000, verbose=False
            )
            pr_map: dict[str, float] = {nid: sc for nid, sc in pr}
            # Keep entities without PR score to avoid over-filtering in sparse graphs
            filtered_entity_ids: list[str] = [
                eid for eid in entity_ids if pr_map.get(eid, 1.0) >= 0.15
            ]

            # Expand to new documents (caps: 8 per entity, 150 total)
            graph_doc_ids: set[str] = set()
            for eid in filtered_entity_ids:
                if len(graph_doc_ids) >= 150:
                    break
                try:
                    doc_neighbors = self.db.graph.get_neighbors(
                        from_node_id_or_label=eid,
                        relation=RelationIsFrom,
                        limit=8,
                        verbose=False,
                    )
                    for from_node, _edge, to_node in doc_neighbors:
                        # Accept either side that is document
                        if to_node.kind == "document":
                            graph_doc_ids.add(to_node.label)
                        if from_node.kind == "document":
                            graph_doc_ids.add(from_node.label)
                        if len(graph_doc_ids) >= 150:
                            break
                except Exception:
                    continue

            new_graph_docs = graph_doc_ids.difference(set(seed_doc_ids))
            if len(new_graph_docs) == 0:
                logger.info(
                    "[gRAG_Iter] No new documents from graph expansion; stopping."
                )
                break

            # Build centroid from candidate document points
            graph_vectors: list[list[float]] = []
            for gdoc in list(new_graph_docs)[:150]:
                try:
                    pts = self.db.points.gen(
                        document_id=gdoc,
                        limit=3,
                        with_embedding=True,
                        verbose=False,
                    )
                    for pt in pts:
                        if pt.embedding:
                            graph_vectors.append(pt.to_python())
                except Exception:
                    continue

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
                embedding_dimensions=self.db_manifest.embedding_dimensions,
                documents_table_name=dvs.DVS_DOCUMENTS_TABLE_NAME,
                points_table_name=dvs.DVS_POINTS_TABLE_NAME,
                conn=self.new_connection(read_only=True),
                with_embedding=False,
                debug=self.v(verbose),
                console=self.settings.console,
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

    async def graph_rag_search_context_aware(
        self,
        query: str,
        top_k: int = 3,
        *,
        context_similarity_threshold: float = 0.7,
        max_expansion_steps: int = 2,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[GraphRAGResult]:
        """Strategy 5: baseline seeds → is_from expand → context-centroid ranking.
        Pros: low-latency; mitigates semantic drift via context filter; explainable.
        Cons: needs embeddings; sparse graphs reduce gains; threshold sensitive.
        Use when: you prefer contextual precision with tight latency and minimal
        global graph prerequisites; hub suppression optional.
        """
        from dvs.types.edge import RelationIsFrom

        # 0) Baseline vector search to get seed context
        baseline_results: list[tuple[Point, Document, float]] = await self.search(
            query=query,
            top_k=max(1, top_k),
            with_embedding=True,  # need embeddings to compute centroids
            verbose=self.v(verbose),
        )

        if not baseline_results:
            return []

        # 1) Build context centroid from seed points
        def mean_vector(vectors: list[list[float]]) -> list[float]:
            if not vectors:
                return []
            length: int = len(vectors[0])
            return [
                sum(v[i] for v in vectors) / float(len(vectors)) for i in range(length)
            ]

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
                    pts = self.db.points.gen(
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

        if self.v(verbose):
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
            if self.v(verbose):
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

        context_centroid: list[float] = mean_vector(seed_vectors)
        if self.v(verbose):
            logger.debug(
                (
                    f"[S5] seed_vecs(after_fetch)={len(seed_vectors)} "
                    + f"centroid_dim={len(context_centroid)}"
                )
            )

        # Query embedding (for vector similarity against candidates)
        query_vector: list[float] = (
            await SearchRequest.to_vectors(
                [SearchRequest(query=query, top_k=1)],
                model=self.model,
                model_settings=self.model_settings,
            )
        )[0]

        # 2) Single-step graph expansion via entities (is_from)
        #    Optionally respect max_expansion_steps>0; here we perform one step.
        if max_expansion_steps <= 0:
            max_expansion_steps = 1

        entity_ids: set[str] = set()
        for _pt, doc, _ in baseline_results:
            try:
                doc_node = self.db.graph.nodes.retrieve_by_label(
                    doc.document_id, verbose=False
                )
                neighbors = self.db.graph.get_neighbors(
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

        if self.v(verbose):
            logger.debug(f"[S5] collected_entities={len(entity_ids)}")

        # 2b) Light hub suppression via PageRank thresholding (optional, conservative)
        try:
            pr = self.db.graph.pagerank(
                relation=RelationIsFrom, limit=5000, verbose=False
            )
            pr_map: dict[str, float] = {nid: sc for nid, sc in pr}
            before_pr: int = len(entity_ids)
            entity_ids = {eid for eid in entity_ids if pr_map.get(eid, 0.1) >= 0.1}
            after_pr: int = len(entity_ids)
            if self.v(verbose):
                logger.debug(
                    (
                        f"[S5] hub_suppress entities before={before_pr} "
                        + f"after={after_pr}"
                    )
                )
        except Exception:
            pass

        # 3) Collect candidate documents from these entities
        candidate_doc_ids: set[str] = set()
        for eid in list(entity_ids)[:300]:
            try:
                doc_neighbors = self.db.graph.get_neighbors(
                    from_node_id_or_label=eid,
                    relation=RelationIsFrom,
                    limit=8,
                    verbose=False,
                )
                for from_node, _edge, to_node in doc_neighbors:
                    if to_node.kind == "document":
                        candidate_doc_ids.add(to_node.label)
                    if from_node.kind == "document":
                        candidate_doc_ids.add(from_node.label)
            except Exception:
                continue

        # Remove seeds from candidates
        candidate_doc_ids.difference_update(set(seed_doc_ids))

        if self.v(verbose):
            logger.debug(
                (
                    f"[S5] candidate_docs={len(candidate_doc_ids)} "
                    + "(after removing seeds)"
                )
            )

        if not candidate_doc_ids:
            # Fallback: return baseline wrapped
            max_score: float = (
                float(baseline_results[0][2]) if baseline_results else 1.0
            )
            if self.v(verbose):
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
        def centroid_for_document(doc_id: str, limit_points: int = 5) -> list[float]:
            vecs: list[list[float]] = []
            try:
                pts = self.db.points.gen(
                    document_id=doc_id,
                    limit=limit_points,
                    with_embedding=True,
                    verbose=False,
                )
                for p in pts:
                    if p.embedding:
                        vecs.append(p.to_python())
            except Exception:
                pass
            return mean_vector(vecs)

        alpha: float = 0.6  # weight for query-vs-candidate vector similarity
        beta: float = 0.4  # weight for context-vs-candidate similarity

        scored: list[tuple[Document, float, float, float]] = []
        log_counter: int = 0
        for doc_id in list(candidate_doc_ids)[:500]:
            try:
                cand_centroid: list[float] = centroid_for_document(doc_id)
                if not cand_centroid:
                    if self.v(verbose) and log_counter < 10:
                        logger.debug(f"[S5] skip doc={doc_id} (no centroid)")
                    continue
                vec_sim: float = float(
                    self._cosine_similarity(query_vector, cand_centroid)
                )
                ctx_sim: float = float(
                    self._cosine_similarity(context_centroid, cand_centroid)
                )
                if ctx_sim < float(context_similarity_threshold):
                    if self.v(verbose) and log_counter < 10:
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
                doc = self.db.documents.retrieve(doc_id, verbose=False)
                scored.append((doc, combined, vec_sim, ctx_sim))
                if self.v(verbose) and log_counter < 10:
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
            if self.v(verbose):
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

    @functools.cached_property
    def db(self) -> "DB":
        from dvs.db.api import DB

        return DB(self)

    @functools.cached_property
    def tokens(self) -> "Tokens":
        from dvs.tokens import Tokens

        return Tokens(self)

    def v(self, verbose: bool | None = None) -> bool:
        """Get verbosity setting, with optional override."""
        return self.verbose if verbose is None else verbose

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _calculate_graph_relevance(
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
            target_node = self.db.graph.nodes.retrieve_by_label(
                target_doc_id, verbose=self.v(verbose)
            )

            for original_doc_id in original_doc_ids:
                try:
                    original_node = self.db.graph.nodes.retrieve_by_label(
                        original_doc_id, verbose=False
                    )

                    # Calculate shortest path distance
                    paths = self.db.graph.get_shortest_paths(
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
