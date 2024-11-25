"""
DVS - DuckDB Vector Similarity Search (VSS) API

This module implements a FastAPI-based web service for performing vector similarity searches
using DuckDB as the backend database and OpenAI's embedding models for vector representation.

The API provides endpoints for single and bulk vector similarity searches, allowing users to
find similar documents or data points based on text queries or pre-computed vector embeddings.

Key Components:
---------------
1. FastAPI Application: Handles HTTP requests and responses.
2. DuckDB Integration: Utilizes DuckDB for efficient storage and querying of vector data.
3. OpenAI Embedding: Converts text queries into vector embeddings using OpenAI's models.
4. Caching: Implements a disk-based cache to store and retrieve computed embeddings.
5. Vector Search: Performs cosine similarity-based searches on the vector database.

Main Endpoints:
---------------
- GET /: Root endpoint providing API status and information.
- POST /search or /s: Perform a single vector similarity search.
- POST /bulk_search or /bs: Perform multiple vector similarity searches in one request.

Configuration:
--------------
The API behavior is controlled by environment variables and the `Settings` class,
which includes database paths, table names, embedding model details, and cache settings.

Data Models:
------------
- Point: Represents a point in the vector space.
- Document: Contains metadata and content information for documents.
- SearchRequest: Defines parameters for a single search request.
- BulkSearchRequest: Represents multiple search requests in a single call.
- SearchResult: Contains the result of a single vector similarity search.
- SearchResponse: Wraps multiple SearchResults for API responses.

Usage Flow:
-----------
1. Client sends a search request (text or vector) to the API.
2. API converts text to vector embedding if necessary (using OpenAI API).
3. Vector search is performed on the DuckDB database.
4. Results are processed and returned to the client.

Performance Considerations:
---------------------------
- Caching of embeddings reduces API calls and improves response times.
- Bulk search endpoint allows for efficient processing of multiple queries.
- DuckDB's columnar storage and vector operations ensure fast similarity computations.

Dependencies:
-------------
- FastAPI: Web framework for building APIs.
- DuckDB: Embeddable SQL OLAP database management system.
- OpenAI: For generating text embeddings.
- Pydantic: Data validation and settings management.
- NumPy: For numerical operations on vectors.

For detailed API documentation, refer to the OpenAPI schema available at the /docs endpoint.

"""  # noqa: E501

from dvs.app_builder import build_app

app = build_app()
