from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from dvs.types.document import Document
from dvs.types.point import Point
from dvs.types.search_result import SearchResult


class SearchResponse(BaseModel):
    """
    Represents the response to a single vector similarity search query.

    This class encapsulates a list of SearchResult objects, providing a
    structured way to return multiple matching results for a given query.

    Attributes
    ----------
    results : List[SearchResult]
        A list of SearchResult objects, each representing a matched item
        from the vector similarity search.

    Methods
    -------
    from_search_results(search_results: List[Tuple[Point, Optional[Document], float]]) -> SearchResponse
        Class method to create a SearchResponse instance from a list of search result tuples.

    Notes
    -----
    The results are typically ordered by relevance score in descending order,
    with the most similar matches appearing first in the list.

    Examples
    --------
    >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
    >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
    >>> result1 = SearchResult(point=point1, document=document1, relevance_score=0.95)
    >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
    >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
    >>> result2 = SearchResult(point=point2, document=document2, relevance_score=0.85)
    >>> response = SearchResponse(results=[result1, result2])
    >>> print(len(response.results))
    2
    """  # noqa: E501

    results: List[SearchResult] = Field(
        default_factory=list,
        description="A list of search results from the vector similarity search.",
    )

    @classmethod
    def from_search_results(
        cls,
        search_results: List[Tuple["Point", Optional["Document"], float]],
    ) -> "SearchResponse":
        """
        Create a SearchResponse instance from a list of search result tuples.

        Parameters
        ----------
        search_results : List[Tuple[Point, Optional[Document], float]]
            A list of tuples, each containing a point, an optional document,
            and a relevance score.

        Returns
        -------
        SearchResponse
            An instance of SearchResponse created from the input list of tuples.

        Examples
        --------
        >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
        >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
        >>> result1 = (point1, document1, 0.95)
        >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
        >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
        >>> result2 = (point2, document2, 0.85)
        >>> response = SearchResponse.from_search_results([result1, result2])
        >>> print(len(response.results))
        2
        """  # noqa: E501

        return cls.model_validate(
            {
                "results": [
                    SearchResult.from_search_result(res) for res in search_results
                ]
            }
        )
