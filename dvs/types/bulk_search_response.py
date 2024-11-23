from typing import List, Optional, Tuple

from pydantic import BaseModel, Field

from dvs.types.document import Document
from dvs.types.point import Point
from dvs.types.search_response import SearchResponse


class BulkSearchResponse(BaseModel):
    """
    Represents the response to a bulk vector similarity search operation.

    This class encapsulates a list of SearchResponse objects, each corresponding
    to a single query in a bulk search request. It provides a structured way to
    return results for multiple queries in a single response.

    Attributes
    ----------
    results : List[SearchResponse]
        A list of SearchResponse objects, each containing the results for
        a single query in the bulk search operation.

    Methods
    -------
    from_bulk_search_results(bulk_search_results: List[List[Tuple[Point, Optional[Document], float]]]) -> BulkSearchResponse
        Class method to create a BulkSearchResponse instance from a list of bulk search result tuples.

    Notes
    -----
    The order of SearchResponse objects in the results list corresponds to
    the order of queries in the original bulk search request.

    Examples
    --------
    >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
    >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
    >>> result1 = SearchResult(point=point1, document=document1, relevance_score=0.95)
    >>> response1 = SearchResponse(results=[result1])
    >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
    >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
    >>> result2 = SearchResult(point=point2, document=document2, relevance_score=0.85)
    >>> response2 = SearchResponse(results=[result2])
    >>> bulk_response = BulkSearchResponse(results=[response1, response2])
    >>> print(len(bulk_response.results))
    2
    """  # noqa: E501

    results: List[SearchResponse] = Field(
        default_factory=list,
        description="A list of search responses, each corresponding to a query in the bulk search.",  # noqa: E501
    )

    @classmethod
    def from_bulk_search_results(
        cls,
        bulk_search_results: List[List[Tuple["Point", Optional["Document"], float]]],
    ) -> "BulkSearchResponse":
        """
        Create a BulkSearchResponse instance from a list of bulk search result tuples.

        Parameters
        ----------
        bulk_search_results : List[List[Tuple[Point, Optional[Document], float]]]
            A list of lists, where each inner list contains tuples of search results
            for a single query in the bulk search operation.

        Returns
        -------
        BulkSearchResponse
            An instance of BulkSearchResponse created from the input list of bulk search results.

        Examples
        --------
        >>> point1 = Point(point_id="1", document_id="doc1", content_md5="abc123", embedding=[0.1, 0.2, 0.3])
        >>> document1 = Document(document_id="doc1", name="Doc 1", content="Content 1")
        >>> result1 = [(point1, document1, 0.95)]
        >>> point2 = Point(point_id="2", document_id="doc2", content_md5="def456", embedding=[0.4, 0.5, 0.6])
        >>> document2 = Document(document_id="doc2", name="Doc 2", content="Content 2")
        >>> result2 = [(point2, document2, 0.85)]
        >>> bulk_response = BulkSearchResponse.from_bulk_search_results([result1, result2])
        >>> print(len(bulk_response.results))
        2
        """  # noqa: E501

        return cls.model_validate(
            {
                "results": [
                    SearchResponse.from_search_results(search_results)
                    for search_results in bulk_search_results
                ]
            }
        )
