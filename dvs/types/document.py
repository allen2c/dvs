import hashlib
import time
from typing import Any, ClassVar, Dict, Optional, Text

from pydantic import BaseModel, Field

from dvs.utils.qs import DocumentQuerySet, DocumentQuerySetDescriptor


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
        ...,
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
    metadata: Optional[Dict[Text, Any]] = Field(
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
    objects: ClassVar["DocumentQuerySetDescriptor"] = DocumentQuerySetDescriptor()

    @classmethod
    def query_set(cls) -> "DocumentQuerySet":
        return DocumentQuerySet(cls)

    @classmethod
    def hash_content(cls, content: Text) -> Text:
        return hashlib.md5(content.strip().encode("utf-8")).hexdigest()

    def strip(self, *, copy: bool = False) -> "Document":
        _doc = self.model_copy(deep=True) if copy else self
        _doc.content = _doc.content.strip()
        new_md5 = self.hash_content(_doc.content)
        if _doc.content_md5 != new_md5:
            _doc.content_md5 = new_md5
            _doc.updated_at = int(time.time())
        return _doc
