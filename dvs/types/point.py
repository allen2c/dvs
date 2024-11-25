from typing import Any, ClassVar, Dict, List, Optional, Text

from pydantic import BaseModel, Field

from dvs.utils.qs import PointQuerySet, PointQuerySetDescriptor


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
        ...,
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

    @property
    def is_embedded(self) -> bool:
        return len(self.embedding) > 0
