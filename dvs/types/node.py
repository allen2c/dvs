import typing

import pydantic
from str_or_none import str_or_none

import dvs.utils.ids


class Node(pydantic.BaseModel):
    node_id: str = pydantic.Field(
        default_factory=lambda: dvs.utils.ids.get_id("n"),
        description="Unique identifier for the node.",
    )
    label: str = pydantic.Field(
        ...,
        description="Label of the node. This is 'document_id' for document node.",
    )
    kind: typing.Literal["entity", "document"] = pydantic.Field(
        ...,
        description="Kind of the node.",
    )
    entity: str = ""

    @pydantic.model_validator(mode="after")
    def validate_label(self) -> typing.Self:
        from dvs.utils.format_string import sanitize_xml_string

        _label = str_or_none(self.label)
        if _label is None:
            raise ValueError("Label is required")
        else:
            self.label = _label

        self.label = sanitize_xml_string(self.label)
        return self

    def __hash__(self) -> int:
        """Make Node hashable using node_id."""
        return hash(self.node_id)

    def __eq__(self, other: object) -> bool:
        """Compare Nodes based on node_id for hash consistency."""
        if not isinstance(other, Node):
            return NotImplemented
        return self.node_id == other.node_id
