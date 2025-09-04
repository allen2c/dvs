import typing

import pydantic
from str_or_none import str_or_none

import dvs.utils.ids

RelationIsA: typing.Literal["is_a"] = "is_a"
RelationHasA: typing.Literal["has_a"] = "has_a"
RelationRelatedTo: typing.Literal["related_to"] = "related_to"
RelationIsFrom: typing.Literal["is_from"] = "is_from"

RelationType: typing.TypeAlias = typing.Literal[
    "is_a", "has_a", "related_to", "is_from"
]


class Edge(pydantic.BaseModel):
    edge_id: str = pydantic.Field(
        default_factory=lambda: dvs.utils.ids.get_id("e"),
        description="Unique identifier for the edge.",
    )
    relation: RelationType = pydantic.Field(
        ...,
        description="Relation of the edge.",
    )
    from_node: str = pydantic.Field(
        ...,
        description="Identifier label name of the from node label.",
    )
    to_node: str = pydantic.Field(
        ...,
        description="Identifier label name of the to node label.",
    )

    @pydantic.model_validator(mode="after")
    def validate_relation(self) -> typing.Self:
        _relation = str_or_none(self.relation)
        if _relation is None:
            raise ValueError("Relation is required")
        else:
            self.relation = _relation  # type: ignore
        return self
