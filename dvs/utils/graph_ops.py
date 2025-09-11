from __future__ import annotations


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """Compute elementwise mean for a list of vectors."""
    if not vectors:
        return []
    length: int = len(vectors[0])
    return [sum(v[i] for v in vectors) / float(len(vectors)) for i in range(length)]
