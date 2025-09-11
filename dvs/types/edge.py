"""
# (Node)-[RelationIsFrom]->(DocumentNode)
#   Example: ('sushi', 'is_from', 'doc-456')  # "sushi" concept is mentioned in document "doc-456"
# (Node)-[RelationIsA]->(Node)
#   Example: ('Tesla Model S', 'is_a', 'electric car')  # "Tesla Model S" is a type of "electric car"
# (Node)-[RelationHasA]->(Node)
#   Example: ('Grand Library', 'has_a', 'reading room')  # "Grand Library" has a "reading room"
# (Node)-[RelationRelatedTo]->(Node)
#   Example: ('machine learning', 'related_to', 'artificial intelligence')  # "machine learning" is related to "artificial intelligence"
"""  # noqa: E501

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
    from_node_id: str = pydantic.Field(
        ...,
        description="Identifier of the from node id.",
    )
    from_node_label: str = pydantic.Field(
        ...,
        description="Label name of the from node.",
    )
    to_node_id: str = pydantic.Field(
        ...,
        description="Identifier of the to node id.",
    )
    to_node_label: str = pydantic.Field(
        ...,
        description="Label name of the to node.",
    )

    @pydantic.model_validator(mode="after")
    def validate_relation(self) -> typing.Self:
        _relation = str_or_none(self.relation)
        if _relation is None:
            raise ValueError("Relation is required")
        else:
            self.relation = _relation  # type: ignore
        return self

    def __hash__(self) -> int:
        """Make Edge hashable using edge_id."""
        return hash(self.edge_id)

    def __eq__(self, other: object) -> bool:
        """Compare Edges based on edge_id for hash consistency."""
        if not isinstance(other, Edge):
            return NotImplemented
        return self.edge_id == other.edge_id
