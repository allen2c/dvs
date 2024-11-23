from typing import List

from pydantic import BaseModel, Field

from dvs.types.search_request import SearchRequest


class BulkSearchRequest(BaseModel):
    """
    Represents a bulk search request for multiple vector similarity searches.

    This class allows users to submit multiple search queries in a single API call,
    which can be more efficient than making separate requests for each query.

    Attributes:
        queries (List[SearchRequest]): A list of SearchRequest objects, each representing
            an individual search query with its own parameters.

    Example:
        >>> bulk_request = BulkSearchRequest(queries=[
        ...     SearchRequest(query="How does AI work?", top_k=5),
        ...     SearchRequest(query="What is machine learning?", top_k=3, with_embedding=True)
        ... ])
        >>> print(bulk_request)
        BulkSearchRequest(queries=[SearchRequest(query='How does AI work?', top_k=5, with_embedding=False), SearchRequest(query='What is machine learning?', top_k=3, with_embedding=True)])

    Note:
        The bulk search functionality allows for efficient processing of multiple queries
        in parallel, which can significantly reduce overall response time compared to
        sequential individual requests.
    """  # noqa: E501

    queries: List[SearchRequest] = Field(
        ...,
        description="A list of search requests to be processed in bulk.",
    )
