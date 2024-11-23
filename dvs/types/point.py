from typing import List, Text

from pydantic import BaseModel, Field


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
        ...,
        description="Vector embedding representation of the point.",
    )
