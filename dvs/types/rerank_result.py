import typing

import pydantic


class Usage(pydantic.BaseModel):
    total_tokens: int


class RerankData(pydantic.BaseModel):
    relevance_score: float
    index: int


class RerankResult(pydantic.BaseModel):
    object: typing.Literal["list"]
    data: list[RerankData]
    model: str
    usage: Usage
