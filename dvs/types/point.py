from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Sequence,
    Set,
    Text,
)

from pydantic import BaseModel, Field
from tqdm import tqdm

import dvs.utils.cache
import dvs.utils.chunk
import dvs.utils.ids
import dvs.utils.qs
from dvs.config import console, settings

if TYPE_CHECKING:
    import diskcache
    from openai import OpenAI

    from dvs.types.document import Document


class Point(BaseModel):
    """
    Represents a point in the vector space, associated with a document.

    This class encapsulates the essential information about a point in the vector space,
    including its unique identifier, the document it belongs to, a content hash, and its
    vector embedding.

    Attributes:
        point_id (Text): A unique identifier for the point in the vector space.
        document_id (Text): The identifier of the document this point is associated with.
        content_md5 (Text): An MD5 hash of the content, used for quick comparisons and integrity checks.
        embedding (List[float]): The vector embedding representation of the point in the vector space.

    The Point class is crucial for vector similarity search operations, as it contains
    the embedding that is used for comparison with query vectors.
    """  # noqa: E501

    point_id: Text = Field(
        default_factory=lambda: dvs.utils.ids.get_id("pt"),
        description="Unique identifier for the point in the vector space.",
    )
    document_id: Text = Field(
        ...,
        description="Identifier of the associated document.",
    )
    content_md5: Text = Field(
        ...,
        description="MD5 hash of the content for quick comparison and integrity checks.",  # noqa: E501
    )
    embedding: List[float] = Field(
        default_factory=list,
        max_length=settings.EMBEDDING_DIMENSIONS,
        description="Vector embedding representation of the point.",
    )
    metadata: Dict[Text, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the point.",
    )

    # Class variables
    objects: ClassVar[dvs.utils.qs.PointQuerySetDescriptor] = (
        dvs.utils.qs.PointQuerySetDescriptor()
    )

    @classmethod
    def query_set(cls) -> dvs.utils.qs.PointQuerySet:
        return dvs.utils.qs.PointQuerySet(cls)

    @classmethod
    def set_embeddings_from_contents(
        cls,
        points: Sequence["Point"],
        contents: Sequence[Text] | Sequence["Document"],
        *,
        openai_client: "OpenAI",
        batch_size: int = 500,
        cache: Optional["diskcache.Cache"] = None,
        debug: bool = False,
    ) -> List["Point"]:
        if len(points) != len(contents):
            raise ValueError("Points and contents must be the same length")

        _iter_chunks = zip(
            dvs.utils.chunk.chunks(points, batch_size),
            dvs.utils.chunk.chunks(contents, batch_size),
        )
        if debug is True:
            _iter_chunks = tqdm(
                _iter_chunks,
                desc="Creating embeddings",
                total=len(points) // batch_size + 1,
                leave=False,
            )

        # Create embeddings
        cache_hits_count = 0
        for batched_points, batched_contents in _iter_chunks:
            # Ensure contents are clean strings
            _contents: List[Text] = [
                (
                    content.strip()
                    if isinstance(content, Text)
                    else content.content.strip()
                )
                for content in batched_contents
            ]

            # Check cache
            _cache_hit: Set[int] = set()
            if cache is not None:
                for _idx, (_point, _content) in enumerate(
                    zip(batched_points, _contents)
                ):
                    _cached_embedding = cache.get(
                        dvs.utils.cache.get_embedding_cache_key(_content)
                    )
                    if _cached_embedding is not None:
                        _point.embedding = _cached_embedding  # type: ignore
                        _cache_hit.add(_idx)
            cache_hits_count += len(_cache_hit)

            # Create embeddings
            _contents_to_embed = [
                c for idx, c in enumerate(_contents) if idx not in _cache_hit
            ]
            if len(_contents_to_embed) > 0:
                emb_res = openai_client.embeddings.create(
                    input=_contents_to_embed,
                    model=settings.EMBEDDING_MODEL,
                    dimensions=settings.EMBEDDING_DIMENSIONS,
                )
                for point, content, embedding in zip(
                    [
                        pt
                        for idx, pt in enumerate(batched_points)
                        if idx not in _cache_hit
                    ],
                    _contents_to_embed,
                    emb_res.data,
                ):
                    point.embedding = embedding.embedding
                    if cache is not None:
                        cache.set(
                            dvs.utils.cache.get_embedding_cache_key(content),
                            embedding.embedding,
                        )

        output = list(points)
        if any(
            pt.is_embedded is False
            or len(pt.embedding) != settings.EMBEDDING_DIMENSIONS
            for pt in output
        ):
            raise ValueError("Not all points were embedded, it is programmer error")
        if debug is True:
            console.print(f"Created {len(output)} embeddings")
            if cache is not None:
                console.print(f"Cache hits: {cache_hits_count} / {len(output)}")
        return output

    @property
    def is_embedded(self) -> bool:
        return len(self.embedding) > 0
