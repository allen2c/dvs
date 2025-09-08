from typing import Literal

import pydantic

from dvs.types.document import Document


class GraphRAGResult(pydantic.BaseModel):
    """Result item for Graph-RAG search without point payload."""

    document: Document
    score: float
    vector_score: float | None = None
    graph_score: float | None = None
    iterations: int | None = None
    rank: int | None = None
    normalized_score: float | None = None
    strategy: (
        Literal[
            "vector_expansion",
            "graph_guided",
            "hybrid_scoring",
            "iterative_refinement",
        ]
        | None
    ) = None
