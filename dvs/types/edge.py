import typing

import pydantic
from str_or_none import str_or_none

import dvs.utils.ids


class Edge(pydantic.BaseModel):
    edge_id: str = pydantic.Field(
        default_factory=lambda: dvs.utils.ids.get_id("e"),
        description="Unique identifier for the edge.",
    )
    label: str = pydantic.Field(
        ...,
        description="Label of the node.",
    )
    from_node: str = pydantic.Field(
        ...,
        description="Identifier label name of the from node.",
    )
    to_node: str = pydantic.Field(
        ...,
        description="Identifier label name of the to node.",
    )
    document_ids: str = pydantic.Field(
        ...,
        description="Identifier of the associated documents separated by comma.",
    )

    @pydantic.model_validator(mode="after")
    def validate_label(self) -> typing.Self:
        _label = str_or_none(self.label)
        if _label is None:
            raise ValueError("Label is required")
        else:
            self.label = _label
        return self

    def merge(self, other: "Edge") -> typing.Self:
        if self.label != other.label:
            raise ValueError("Labels do not match")

        _ids = set(self.document_ids.split(",")) | set(other.document_ids.split(","))
        self.document_ids = ",".join(sorted(_ids))
        return self
