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
        description="Label of the node.",
    )
    kind: typing.Literal["entity", "document"] = pydantic.Field(
        ...,
        description="Kind of the node.",
    )
    entity: str = ""

    @pydantic.model_validator(mode="after")
    def validate_label(self) -> typing.Self:
        _label = str_or_none(self.label)
        if _label is None:
            raise ValueError("Label is required")
        else:
            self.label = _label
        return self
