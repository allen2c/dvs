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

        self.db_manifest = self._ensure_manifest(
            self.model, self.model_settings, verbose=self.verbose
        )

        self.db.touch(verbose=self.verbose)

    @property
    def duckdb_path(self) -> pathlib.Path:
        return self.settings.duckdb_path

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        """
        Always open a new duckdb connection.
        """
        return duckdb.connect(self.duckdb_path)

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
        create_points_batch_size: int = 100,
        ignore_same_content: bool = True,
        lines_per_chunk: int = 20,
        tokens_per_chunk: int = 500,
        verbose: bool | None = None,
        very_verbose: bool | None = None,
    ) -> typing.Dict:
        """
        Add one or more documents to the vector similarity search database.
        Processes documents, generates embeddings via OpenAI API, and stores in DuckDB.
        Returns dict with creation stats and ignores duplicates if ignore_same_content=True.
        """  # noqa: E501

        verbose = self.verbose if verbose is None else verbose
        very_verbose = True if very_verbose else False

        # Validate documents
        docs: list["Document"] = Document.from_contents(documents)
        ignored_docs_indexes: list[int] = []
        creating_points: list[Point] = []
        creating_point_contents: list[str] = []

        # Chunk documents
        chunked_docs = [
            chunked_doc
            for doc in docs
            for chunked_doc in doc.to_chunked_documents(
                lines_per_chunk=lines_per_chunk,
                tokens_per_chunk=tokens_per_chunk,
                encoding=self.tokens.enc,
            )
        ]

        # Collect documents
        for idx, doc in tqdm(
            enumerate(chunked_docs),
            total=len(chunked_docs),
            disable=not verbose,
            desc="Checking for duplicate documents",
        ):
            if ignore_same_content:
                if self.db.documents.content_exists(doc.content_md5, verbose=False):
                    if very_verbose:
                        logger.debug(
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

        # Collect points
        for doc in creating_docs:
            points_with_contents: tuple[list[Point], list[str]] = (
                doc.to_points_with_contents(with_embeddings=False)
            )
            creating_points.extend(points_with_contents[0])
            creating_point_contents.extend(points_with_contents[1])

        # Create documents into the database
        self.db.documents.bulk_create(creating_docs, verbose=verbose)

        # Create embeddings (assign embeddings to points in place)
        for batch_points_with_contents in chunks(
            zip(creating_points, creating_point_contents),
            batch_size=create_points_batch_size,
        ):
            Point.set_embeddings_from_contents(
                [pt for pt, _ in batch_points_with_contents],
                [c for _, c in batch_points_with_contents],
                model=self.model,
                model_settings=self.model_settings,
            )
            self.db.points.bulk_create(
                [pt for pt, _ in batch_points_with_contents],
                batch_size=create_points_batch_size,
                verbose=verbose,
            )

        return {
            "success": True,
            "created_documents": len(creating_docs),
            "ignored_documents": len(ignored_docs_indexes),
            "created_points": len(creating_points),
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

    @functools.cached_property
    def tokens(self) -> "Tokens":
        from dvs.tokens import Tokens

        return Tokens(self)

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
        if dvs.MANIFEST_TABLE_NAME not in self.db.show_table_names():
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
