import json
import time
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Text

from pydantic import BaseModel, Field

import dvs.utils.hash
import dvs.utils.ids
import dvs.utils.qs
from dvs.config import console, settings

if TYPE_CHECKING:
    from openai import OpenAI

    from dvs.types.point import Point


class Document(BaseModel):
    """
    Represents a document in the system, containing metadata and content information.

    This class encapsulates all the relevant information about a document, including
    its unique identifier, name, content, and various metadata fields. It is designed
    to work in conjunction with the Point class for vector similarity search operations.

    Attributes:
        document_id (Text): A unique identifier for the document.
        name (Text): The name or title of the document.
        content (Text): The full text content of the document.
        content_md5 (Text): An MD5 hash of the content for integrity checks.
        metadata (Optional[Dict[Text, Any]]): Additional metadata associated with the document.
        created_at (Optional[int]): Unix timestamp of when the document was created.
        updated_at (Optional[int]): Unix timestamp of when the document was last updated.

    The Document class is essential for storing and retrieving document information
    in the vector similarity search system. It provides a structured way to manage
    document data and metadata, which can be used in conjunction with vector embeddings
    for advanced search and retrieval operations.
    """  # noqa: E501

    document_id: Text = Field(
        default_factory=lambda: dvs.utils.ids.get_id("doc"),
        description="Unique identifier for the document.",
    )
    name: Text = Field(
        ...,
        description="Name or title of the document.",
    )
    content: Text = Field(
        ...,
        description="Full text content of the document.",
    )
    content_md5: Text = Field(
        ...,
        description="MD5 hash of the content for integrity checks.",
    )
    metadata: Dict[Text, Any] = Field(
        default_factory=dict,
        description="Additional metadata associated with the document.",
    )
    created_at: Optional[int] = Field(
        default=None,
        description="Unix timestamp of document creation.",
    )
    updated_at: Optional[int] = Field(
        default=None,
        description="Unix timestamp of last document update.",
    )

    # Class variables
    objects: ClassVar[dvs.utils.qs.DocumentQuerySetDescriptor] = (
        dvs.utils.qs.DocumentQuerySetDescriptor()
    )

    @classmethod
    def query_set(cls) -> dvs.utils.qs.DocumentQuerySet:
        return dvs.utils.qs.DocumentQuerySet(cls)

    @classmethod
    def hash_content(cls, content: Text) -> Text:
        return dvs.utils.hash.hash_content(content)

    def strip(self, *, copy: bool = False) -> "Document":
        _doc = self.model_copy(deep=True) if copy else self
        _doc.content = _doc.content.strip()
        new_md5 = self.hash_content(_doc.content)
        if _doc.content_md5 != new_md5:
            _doc.content_md5 = new_md5
            _doc.updated_at = int(time.time())
        return _doc

    def to_points(
        self,
        *,
        openai_client: Optional["OpenAI"] = None,
        with_embeddings: bool = False,
        metadata: Optional[Dict[Text, Any]] = None,
        debug: bool = False,
    ) -> List["Point"]:
        """"""

        from dvs.types.point import Point

        self.strip()
        _meta = json.loads(json.dumps(metadata or {}, default=str))
        _pt_data = {
            "point_id": dvs.utils.ids.get_id("pt"),
            "document_id": self.document_id,
            "content_md5": self.content_md5,
            "metadata": _meta,
        }
        _pt = Point.model_validate(_pt_data)
        if with_embeddings:
            if openai_client is None:
                raise ValueError(
                    "OpenAI client is required when `with_embeddings` is True"
                )
            _pt.embedding = (
                openai_client.embeddings.create(
                    input=self.content,
                    model=settings.EMBEDDING_MODEL,
                    dimensions=settings.EMBEDDING_DIMENSIONS,
                )
                .data[0]
                .embedding
            )
        _pts = [_pt]

        if debug:
            console.print(f"Created {len(_pts)} points")
        return _pts
