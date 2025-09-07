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
        """
        Add one or more documents to the vector similarity search database.
        Processes documents, generates embeddings via OpenAI API, and stores in DuckDB.
        Returns dict with creation stats and ignores duplicates if ignore_same_content=True.
        """  # noqa: E501

        verbose = self.verbose if verbose is None else verbose

        # Validate documents
        docs: list["Document"] = Document.from_contents(documents)
        ignored_docs_indexes: list[int] = []
        creating_points_count: int = 0

        # Chunk documents
        chunked_docs = [
            chunked_doc
            for doc in tqdm(
                docs, total=len(docs), disable=not verbose, desc="Chunking documents"
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
        """
        Remove one or more documents and their associated vector points from the database.
        Accepts single document ID or iterable of IDs and deletes both documents and points.
        Operation is irreversible and raises NotFoundError if document ID doesn't exist.
        """  # noqa: E501
        verbose = self.verbose if verbose is None else verbose
        doc_ids = [doc_ids] if isinstance(doc_ids, str) else list(doc_ids)

        self.db.points.remove_many(document_ids=doc_ids, verbose=verbose)
        for doc_id in doc_ids:
            self.db.documents.remove(doc_id, verbose=verbose)

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

        verbose = self.verbose if verbose is None else verbose

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
    ) -> list[tuple["Point", "Document", float]]:
        """
        Graph-RAG Strategy 1: Vector + Graph Expansion

        Original query → Vector search → Find relevant nodes → Graph expansion → Re-scoring

        Args:
            query: Search query
            top_k: Number of results to return
            graph_expansion_depth: Graph expansion depth
            vector_weight: Vector similarity weight
            graph_weight: Graph relationship weight
            with_embedding: Whether to include embedding
            verbose: Whether to show detailed information

        Returns:
            Search results list [(Point, Document, relevance_score), ...]
        """  # noqa: E501
        verbose = self.verbose if verbose is None else verbose

        # Step 1: Vector search - Find most relevant documents
        initial_results = await self.search(
            query=query,
            top_k=top_k * 2,  # Expand candidate set
            with_embedding=with_embedding,
            verbose=verbose,
        )

        if not initial_results:
            return []

        # Step 2: Find corresponding document nodes from document IDs
        document_ids = [doc.document_id for _, doc, _ in initial_results]
        document_nodes = []

        for doc_id in document_ids:
            try:
                # Try to find corresponding document node (label = document_id)
                doc_node = self.db.graph.nodes.retrieve(doc_id, verbose=False)
                document_nodes.append(doc_node)
            except Exception:
                # Skip if corresponding node not found
                continue

        # Step 3: Find related entity nodes through "is_from" relationship
        related_entities = set()
        for doc_node in document_nodes:
            try:
                # Find all entity nodes connected to this document
                neighbors = self.db.graph.get_neighbors(
                    from_node_id_or_label=doc_node.node_id,
                    relation="is_from",
                    limit=10,
                    verbose=False,
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
                    neighbors = self.db.graph.get_neighbors(
                        to_node_id_or_label=entity_id,
                        relation="is_from",
                        limit=5,
                        verbose=False,
                    )

                    # Collect new document IDs
                    for _, _, doc_node in neighbors:
                        if doc_node.kind == "document":
                            expanded_document_ids.add(doc_node.node_id)
                except Exception:
                    continue

        # Step 5: Re-vector search on expanded document collection
        expanded_candidates = []
        for doc_id in expanded_document_ids:
            try:
                # Get all points for the document
                points = self.db.points.gen(document_id=doc_id, limit=10)
                for point in points:
                    if point.embedding:
                        expanded_candidates.append(point)
            except Exception:
                continue

        # Fall back to original results if no candidates found
        if not expanded_candidates:
            return initial_results[:top_k]

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
        final_results = []
        for point, vector_score in scored_candidates:
            try:
                doc = self.db.documents.retrieve(point.document_id, verbose=False)

                # Calculate graph relevance: based on distance to original
                # query-related documents
                graph_score = self._calculate_graph_relevance(
                    point.document_id, document_ids, related_entities
                )

                # Combined scoring
                combined_score = (
                    vector_weight * vector_score + graph_weight * graph_score
                )

                final_results.append((point, doc, combined_score))

            except Exception:
                continue

        # Step 8: Sort by combined score and return top-k
        final_results.sort(key=lambda x: x[2], reverse=True)
        return final_results[:top_k]

    async def graph_rag_search_graph_guided(
        self,
        query: str,
        top_k: int = 3,
        *,
        relation_types: list[str] | None = None,
        centrality_threshold: float = 0.5,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[tuple["Point", "Document", float]]:
        """
        Graph-RAG Strategy 2: Graph-Guided Vector Search

        Query → Graph search popular nodes → Vector search on these nodes → Combined scoring  # noqa

        Args:
            query: Search query
            top_k: Number of results to return
            relation_types: Graph relation type filters
            centrality_threshold: Centrality threshold
            with_embedding: Whether to include embedding
            verbose: Whether to show detailed information

        Returns:
            Search results list [(Point, Document, relevance_score), ...]
        """
        # TODO: Implement Strategy 2 - Graph-Guided Vector Search
        raise NotImplementedError("Strategy 2 not yet implemented")

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
    ) -> list[tuple["Point", "Document", float]]:
        """
        Graph-RAG Strategy 3: Hybrid Scoring

        Vector similarity + Graph importance + Graph distance = Combined scoring

        Args:
            query: Search query
            top_k: Number of results to return
            vector_weight: Vector similarity weight
            graph_importance_weight: Graph importance weight
            graph_distance_weight: Graph distance weight
            with_embedding: Whether to include embedding
            verbose: Whether to show detailed information

        Returns:
            Search results list [(Point, Document, relevance_score), ...]
        """
        # TODO: Implement Strategy 3 - Hybrid Scoring
        raise NotImplementedError("Strategy 3 not yet implemented")

    async def graph_rag_search_iterative_refinement(
        self,
        query: str,
        top_k: int = 3,
        *,
        max_iterations: int = 3,
        refinement_threshold: float = 0.1,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[tuple["Point", "Document", float]]:
        """
        Graph-RAG Strategy 4: Iterative Refinement

        Initial search → Expand query based on results → Search again → Repeat until convergence  # noqa

        Args:
            query: Search query
            top_k: Number of results to return
            max_iterations: Maximum number of iterations
            refinement_threshold: Convergence threshold
            with_embedding: Whether to include embedding
            verbose: Whether to show detailed information

        Returns:
            Search results list [(Point, Document, relevance_score), ...]
        """
        # TODO: Implement Strategy 4 - Iterative Refinement
        raise NotImplementedError("Strategy 4 not yet implemented")

    async def graph_rag_search_context_aware(
        self,
        query: str,
        top_k: int = 3,
        *,
        context_similarity_threshold: float = 0.7,
        max_expansion_steps: int = 2,
        with_embedding: bool = False,
        verbose: bool | None = None,
    ) -> list[tuple["Point", "Document", float]]:
        """
        Graph-RAG Strategy 5: Context-Aware Expansion

        Vector search → Analyze result relationships → Expand search scope based on relationships
        → Filter irrelevant results  # noqa

        Args:
            query: Search query
            top_k: Number of results to return
            context_similarity_threshold: Context similarity threshold
            max_expansion_steps: Maximum expansion steps
            with_embedding: Whether to include embedding
            verbose: Whether to show detailed information

        Returns:
            Search results list [(Point, Document, relevance_score), ...]
        """
        # TODO: Implement Strategy 5 - Context-Aware Expansion
        raise NotImplementedError("Strategy 5 not yet implemented")

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
    ) -> float:
        """Calculate graph relevance for a document"""
        # Give higher score if document is in original search results
        if target_doc_id in original_doc_ids:
            return 1.0

        # Calculate graph distance to original documents
        min_distance = float("inf")

        try:
            # Try to find target document node
            target_node = self.db.graph.nodes.retrieve(target_doc_id, verbose=False)

            for original_doc_id in original_doc_ids:
                try:
                    original_node = self.db.graph.nodes.retrieve(
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
