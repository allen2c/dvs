import functools
import pathlib
import typing

import duckdb
import openai
import openai_embeddings_model as oai_emb_model
from str_or_none import str_or_none

import dvs
import dvs.utils.vss as VSS
from dvs.config import Settings
from dvs.types.document import Document
from dvs.types.point import Point
from dvs.types.search_request import SearchRequest
from dvs.utils.chunk import chunks

if typing.TYPE_CHECKING:
    from dvs.db.api import DB
    from dvs.types.manifest import Manifest as ManifestType


class DVS:
    def __init__(
        self,
        settings: typing.Union[pathlib.Path, str] | Settings,
        *,
        model_settings: oai_emb_model.ModelSettings | None = None,
        model: oai_emb_model.OpenAIEmbeddingsModel | str,
        verbose: bool | None = None,
    ):
        self.settings = self._ensure_dvs_settings(settings)
        self.verbose = verbose or False
        self.model = self._ensure_model(model)
        self.model_settings = model_settings or oai_emb_model.ModelSettings()

        self.db.touch(verbose=verbose)

        self.db_manifest = self._ensure_manifest(
            self.model, self.model_settings, verbose=self.verbose
        )

    @property
    def db_path(self) -> pathlib.Path:
        return pathlib.Path(self.settings.DUCKDB_PATH)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """
        Always open a new duckdb connection.
        """
        return duckdb.connect(self.db_path)

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
        embeddings_batch_size: int = 500,
        create_embeddings_batch_size: int = 100,
        verbose: bool | None = None,
    ) -> typing.Dict:
        """
        Add one or more documents to the vector similarity search database.

        This method processes input documents (either as raw text or Document objects),
        generates vector embeddings using OpenAI's API, and stores both documents and
        their vector points in DuckDB for similarity searching.

        Notes
        -----
        - Input documents are automatically stripped of whitespace
        - Empty documents will raise ValueError
        - For text inputs, document name is derived from first line (truncated to 28 chars)
        - Embeddings are cached to improve performance on repeated content
        - Documents and points are created in bulk transactions for efficiency

        Examples
        --------
        >>> dvs = DVS()
        >>> # Add single document
        >>> dvs.add("This is a sample document")
        >>> # Add multiple documents
        >>> docs = [
        ...     "First document content",
        ...     Document(name="doc2", content="Second document")
        ... ]
        >>> dvs.add(docs)

        Warnings
        --------
        - Large batches of documents may take significant time due to embedding generation
        - OpenAI API costs apply for generating embeddings
        """  # noqa: E501

        verbose = self.verbose if verbose is None else verbose
        output: list[tuple[Document, list[Point]]] = []

        # Validate documents
        docs: list["Document"] = Document.from_contents(documents)
        all_points: list[Point] = []
        all_point_contents: list[str] = []

        # Collect documents and points
        for doc in docs:
            points_with_contents: tuple[list[Point], list[str]] = (
                doc.to_points_with_contents(with_embeddings=False)
            )
            output.append((doc, points_with_contents[0]))
            all_points.extend(points_with_contents[0])
            all_point_contents.extend(points_with_contents[1])

        # Create documents into the database
        self.db.documents.bulk_create(docs, verbose=verbose)

        # Create embeddings (assign embeddings to points in place)
        for batch_points_with_contents in chunks(
            zip(all_points, all_point_contents), batch_size=embeddings_batch_size
        ):
            Point.set_embeddings_from_contents(
                [pt for pt, _ in batch_points_with_contents],
                [c for _, c in batch_points_with_contents],
                model=self.model,
                model_settings=self.model_settings,
            )
            self.db.points.bulk_create(
                [pt for pt, _ in batch_points_with_contents],
                batch_size=create_embeddings_batch_size,
                verbose=verbose,
            )

        return {
            "success": True,
            "created_documents": len(docs),
            "created_points": len(all_points),
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

        This method deletes the specified documents and all their corresponding vector points
        from the DuckDB database. It ensures that both the document data and associated
        vector embeddings are properly cleaned up.

        Notes
        -----
        - Accepts either a single document ID or an iterable of document IDs
        - Removes both document metadata and associated vector points
        - Operations are performed sequentially for each document ID

        Examples
        --------
        >>> dvs = DVS()
        >>> # Remove single document
        >>> dvs.remove("doc-123abc")
        >>> # Remove multiple documents
        >>> dvs.remove(["doc-123abc", "doc-456def"])

        Warnings
        --------
        - This operation is irreversible and will permanently delete the documents
        - If a document ID doesn't exist, a NotFoundError will be raised
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
        Perform an asynchronous vector similarity search using text query.

        This method converts the input text query into a vector embedding using OpenAI's API,
        then searches for similar documents in the DuckDB database using cosine similarity.
        Results are returned as tuples containing the matched point, associated document,
        and relevance score.

        Notes
        -----
        - Query text is automatically stripped of whitespace
        - Empty queries will raise ValueError
        - Embeddings are cached to improve performance on repeated queries
        - Results are ordered by descending relevance score (cosine similarity)

        Examples
        --------
        >>> dvs = DVS()
        >>> results = await dvs.search(
        ...     query="What is machine learning?",
        ...     top_k=3,
        ...     with_embedding=False
        ... )
        >>> for point, document, score in results:
        ...     print(f"Score: {score:.3f}, Doc: {document.name}")

        Warnings
        --------
        - OpenAI API costs apply for generating query embeddings
        - Large top_k values may impact performance
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
            documents_table_name=dvs.DOCUMENTS_TABLE_NAME,
            points_table_name=dvs.POINTS_TABLE_NAME,
            conn=self.conn,
            with_embedding=search_req.with_embedding,
            console=self.settings.console,
        )

        return results

    @functools.cached_property
    def db(self) -> "DB":
        from dvs.db.api import DB

        return DB(self)

    def _ensure_dvs_settings(
        self, settings: typing.Union[pathlib.Path, str] | Settings
    ) -> Settings:
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
        if isinstance(model, oai_emb_model.OpenAIEmbeddingsModel):
            return model
        else:
            return oai_emb_model.OpenAIEmbeddingsModel(
                model, openai.OpenAI(), oai_emb_model.get_default_cache()
            )

    def _ensure_manifest(
        self,
        model: oai_emb_model.OpenAIEmbeddingsModel,
        model_settings: oai_emb_model.ModelSettings,
        verbose: bool,
    ) -> "ManifestType":
        """
        Ensure the manifest of the database is consistent with the model and model settings.

        Set dimensions to model settings if None in place.
        """  # noqa: E501

        manifest: "ManifestType"
        might_manifest = self.db.manifest.receive(verbose=verbose)
        if might_manifest is None:
            if model_settings.dimensions is None:
                raise ValueError(
                    "Could not infer the embedding dimensions, "
                    + "please provide the model settings."
                )
            else:
                manifest = self.db.manifest.create(
                    ManifestType(
                        embedding_model=self.model.model,
                        embedding_dimensions=model_settings.dimensions,
                    ),
                    verbose=verbose,
                )
        else:
            manifest = might_manifest
            if manifest.embedding_model != model.model:
                raise ValueError(
                    "The indicated embedding model is not the same as "
                    + "the one in the manifest of the database"
                )
            if model_settings.dimensions is not None:
                if manifest.embedding_dimensions != model_settings.dimensions:
                    raise ValueError(
                        "The indicated embedding dimensions are not the same as "
                        + "the one in the manifest of the database"
                    )
            else:
                model_settings.dimensions = manifest.embedding_dimensions

        return manifest
