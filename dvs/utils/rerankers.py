import os

import httpx
from str_or_none import str_or_none

from dvs.types.rerank_result import RerankResult


async def rerank_by_voyage(
    query: str, documents: list[str], *, model: str = "rerank-2.5-lite"
) -> RerankResult:
    api_key = str_or_none(os.getenv("VOYAGE_API_KEY"))
    if api_key is None:
        raise ValueError("VOYAGE_API_KEY is not set")

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.voyageai.com/v1/rerank",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            json={"query": query, "documents": documents, "model": model},
        )
        return RerankResult.model_validate(response.json())
