import typing

from ner_agent import Triplet as NerTriplet


class Triplet(NerTriplet):
    relation: typing.Literal["is_a", "has_a", "related_to", "is_from"]
    document_id: str
