# dvs/db/graph/nodes/api.py
import logging
import typing

import dvs
from dvs.types.node import Node as NodeType
from dvs.types.paginations import Pagination

logger = logging.getLogger(__name__)


class Nodes:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        raise NotImplementedError

    def retrieve(
        self, node_id: typing.Text, *, verbose: bool | None = None
    ) -> NodeType:
        raise NotImplementedError

    def create(
        self,
        node: typing.Union[NodeType, typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> NodeType:
        raise NotImplementedError

    def bulk_create(
        self, nodes: typing.Sequence[NodeType], *, verbose: bool | None = None
    ) -> typing.List[NodeType]:
        raise NotImplementedError

    def list(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[NodeType]:
        raise NotImplementedError

    def gen(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[NodeType, None, None]:
        raise NotImplementedError

    def count(self, *, verbose: bool | None = None) -> int:
        raise NotImplementedError
