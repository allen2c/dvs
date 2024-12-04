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
        self.openai_client = openai_client or OpenAI(api_key=settings.OPENAI_API_KEY)
        self.cache = cache or diskcache.Cache(
            settings.CACHE_PATH, size_limit=settings.CACHE_SIZE_LIMIT
        )

        if touch:
            self.touch(raise_if_exists=raise_if_exists, debug=debug)

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(self._db_path)  # Always open a new duckdb connection

    def touch(self, *, raise_if_exists: bool = False, debug: Optional[bool] = None):
        """"""

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
        """"""

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
        """"""

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
        """"""

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
