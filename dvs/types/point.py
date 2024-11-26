from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Sequence, Text

from pydantic import BaseModel, Field
from tqdm import tqdm

import dvs.utils.chunk
import dvs.utils.ids
from dvs.config import settings
from dvs.utils.qs import PointQuerySet, PointQuerySetDescriptor

if TYPE_CHECKING:
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
        description="Vector embedding representation of the point.",
    )
    metadata: Optional[Dict[Text, Any]] = Field(
        default_factory=dict,
        description="Additional metadata associated with the point.",
    )

    # Class variables
    objects: ClassVar["PointQuerySetDescriptor"] = PointQuerySetDescriptor()

    @classmethod
    def query_set(cls) -> "PointQuerySet":
        return PointQuerySet(cls)

    @classmethod
    def set_embeddings_from_contents(
        cls,
        points: Sequence["Point"],
        contents: Sequence[Text] | Sequence["Document"],
        *,
        openai_client: "OpenAI",
        batch_size: int = 500,
        debug: bool = False,
    ) -> Sequence["Point"]:
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
        for batched_points, batched_contents in _iter_chunks:
            _contents: List[Text] = [
                (
                    content.strip()
                    if isinstance(content, Text)
                    else content.content.strip()
                )
                for content in batched_contents
            ]
            emb_res = openai_client.embeddings.create(
                input=_contents,
                model=settings.EMBEDDING_MODEL,
                dimensions=settings.EMBEDDING_DIMENSIONS,
            )
            if len(emb_res.data) != len(batched_points):
                raise ValueError(
                    "Embedding response and points must be the same length"
                )
            for point, embedding in zip(batched_points, emb_res.data):
                point.embedding = embedding.embedding

        return points

    @property
    def is_embedded(self) -> bool:
        return len(self.embedding) > 0
