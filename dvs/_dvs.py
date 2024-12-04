import time
from pathlib import Path
from typing import Iterable, List, Optional, Text, Tuple, Union

import diskcache
import duckdb
from openai import OpenAI

import dvs.utils.vss as VSS
from dvs.config import settings
from dvs.types.document import Document
from dvs.types.point import Point
from dvs.types.search_request import SearchRequest


class DVS:
    def __init__(
        self,
        duckdb_path: Optional[Path] = None,
        *,
        touch: bool = True,
        raise_if_exists: bool = False,
        debug: bool = False,
        openai_client: Optional["OpenAI"] = None,
        cache: Optional["diskcache.Cache"] = None,
    ):
        self._db_path = duckdb_path or Path(settings.DUCKDB_PATH)
        self.debug = debug
        self.openai_client = openai_client or settings.openai_client
        self.cache = cache or settings.cache

        if touch:
            self.touch(raise_if_exists=raise_if_exists, debug=debug)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path)  # Always open a new duckdb connection

    def touch(self, *, raise_if_exists: bool = False, debug: Optional[bool] = None):
        """
        Initialize the DuckDB database tables required for vector similarity search.

        This method creates the necessary database tables (documents and points) with proper
        schemas and indexes. It installs required DuckDB extensions and sets up HNSW indexing
        for efficient vector similarity searches.

        Notes
        -----
        - Creates 'documents' table for storing document metadata and content
        - Creates 'points' table for storing vector embeddings with HNSW indexing
        - Installs required DuckDB extensions (e.g., JSON, httpfs)
        - Sets up indexes for optimized query performance

        Examples
        --------
        >>> dvs = DVS(duckdb_path="./data/vectors.duckdb")
        >>> dvs.touch(raise_if_exists=True, debug=True)

        Warnings
        --------
        If raise_if_exists=True and tables already exist, raises ConflictError
        with status code 409.
        """  # noqa: E501

        debug = self.debug if debug is None else debug

        Document.objects.touch(
            conn=self.conn, raise_if_exists=raise_if_exists, debug=debug
        )
        Point.objects.touch(
            conn=self.conn, raise_if_exists=raise_if_exists, debug=debug
        )

    def add(
        self,
        documents: Union[
            Document,
            Iterable[Document],
            Text,
            Iterable[Text],
            Iterable[Union[Document, Text]],
        ],
        *,
        debug: Optional[bool] = None,
    ) -> List[Tuple[Document, List[Point]]]:
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

        debug = self.debug if debug is None else debug
        output: List[Tuple[Document, List[Point]]] = []

        # Validate documents
        if isinstance(documents, Text) or isinstance(documents, Document):
            documents = [documents]
        docs: List["Document"] = []
        for idx, doc in enumerate(documents):
            if isinstance(doc, Text):
                doc = doc.strip()
                if not doc:
                    raise ValueError(f"Document [{idx}] content cannot be empty: {doc}")
                doc = Document.model_validate(
                    {
                        "name": doc.split("\n")[0][:28],
                        "content": doc,
                        "content_md5": Document.hash_content(doc),
                        "metadata": {
                            "content_length": len(doc),
                        },
                        "created_at": int(time.time()),
                        "updated_at": int(time.time()),
                    }
                )
                doc = doc.strip()
                docs.append(doc)
            else:
                doc = doc.strip()
                if not doc.content.strip():
                    raise ValueError(
                        f"Document [{idx}] content cannot be empty: {doc.content}"
                    )
                docs.append(doc)

        # Collect documents and points
        for doc in docs:
            points: List[Point] = doc.to_points()
            output.append((doc, points))

        # Create embeddings
        all_points = [pt for _, pts in output for pt in pts]
        all_points = Point.set_embeddings_from_contents(
            all_points,
            docs,
            openai_client=self.openai_client,
            cache=self.cache,
            debug=debug,
        )

        # Bulk create documents and points
        docs = Document.objects.bulk_create(docs, conn=self.conn, debug=debug)
        all_points = Point.objects.bulk_create(all_points, conn=self.conn, debug=debug)

        return output

    def remove(
        self,
        doc_ids: Union[Text, Iterable[Text]],
        *,
        debug: Optional[bool] = None,
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

        debug = self.debug if debug is None else debug
        doc_ids = [doc_ids] if isinstance(doc_ids, Text) else list(doc_ids)

        for doc_id in doc_ids:
            Document.objects.remove(doc_id, conn=self.conn, debug=debug)
            Point.objects.remove_many(
                document_ids=[doc_id], conn=self.conn, debug=debug
            )

        return None

    async def search(
        self,
        query: Text,
        top_k: int = 3,
        *,
        with_embedding: bool = False,
        debug: Optional[bool] = None,
    ) -> List[Tuple["Point", Optional["Document"], float]]:
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

        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")

        # Validate search request
        search_req = SearchRequest.model_validate(
            {"query": query, "top_k": top_k, "with_embedding": with_embedding}
        )
        vectors = await SearchRequest.to_vectors(
            [search_req],
            cache=self.cache,
            openai_client=self.openai_client,
        )
        vector = vectors[0]

        # Perform vector search
        results = await VSS.vector_search(
            vector=vector,
            top_k=search_req.top_k,
            conn=self.conn,
            with_embedding=search_req.with_embedding,
        )

        return results
