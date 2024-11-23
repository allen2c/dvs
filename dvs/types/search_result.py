from typing import Optional, Tuple

from pydantic import BaseModel, Field

from dvs.types.document import Document
from dvs.types.point import Point


class SearchResult(BaseModel):
    """
    Represents a single result from a vector similarity search operation.

    This class encapsulates the information returned for each matching item
    in a vector similarity search, including the matched point, its associated
    document (if any), and the relevance score indicating how closely it matches
    the query.

    Attributes
    ----------
    point : Point
        The matched point in the vector space, containing embedding and metadata.
    document : Optional[Document]
        The associated document for the matched point, if available.
    relevance_score : float
        A score indicating the similarity between the query and the matched point,
        typically ranging from 0 to 1, where 1 indicates a perfect match.

    Methods
    -------
    from_search_result(search_result: Tuple[Point, Optional[Document], float]) -> SearchResult
        Class method to create a SearchResult instance from a tuple of search results.

    Notes
    -----
    The relevance score is typically calculated using cosine similarity between
    the query vector and the point's embedding vector.

    Examples
    --------
    >>> point = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
    >>> document = Document(document_id="doc1", name="Example Doc", content="Sample content")
    >>> result = SearchResult(point=point, document=document, relevance_score=0.95)
    >>> print(result.relevance_score)
    0.95
    """  # noqa: E501

    point: Point = Field(
        ...,
        description="The matched point in the vector space.",
    )
    document: Optional[Document] = Field(
        default=None,
        description="The associated document for the matched point, if available.",
    )
    relevance_score: float = Field(
        default=0.0,
        description="The similarity score between the query and the matched point.",
    )

    @classmethod
    def from_search_result(
        cls, search_result: Tuple["Point", Optional["Document"], float]
    ) -> "SearchResult":
        """
        Create a SearchResult instance from a tuple of search results.

        Parameters
        ----------
        search_result : Tuple[Point, Optional[Document], float]
            A tuple containing the point, document, and relevance score.

        Returns
        -------
        SearchResult
            An instance of SearchResult created from the input tuple.

        Examples
        --------
        >>> point = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
        >>> document = Document(document_id="doc1", name="Example Doc", content="Sample content")
        >>> result_tuple = (point, document, 0.95)
        >>> search_result = SearchResult.from_search_result(result_tuple)
        >>> print(search_result.relevance_score)
        0.95
        """  # noqa: E501

        return cls.model_validate(
            {
                "point": search_result[0],
                "document": search_result[1],
                "relevance_score": search_result[2],
            }
        )
