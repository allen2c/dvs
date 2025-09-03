# dvs/db/graph/edges/api.py
import logging
import typing

import dvs
from dvs.types.edge import Edge as EdgeType
from dvs.types.paginations import Pagination

logger = logging.getLogger(__name__)


class Edges:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        raise NotImplementedError

    def retrieve(
        self, edge_id: typing.Text, *, verbose: bool | None = None
    ) -> EdgeType:
        raise NotImplementedError

    def create(
        self,
        edge: typing.Union[EdgeType, typing.Dict],
        *,
        verbose: bool | None = None,
    ) -> EdgeType:
        raise NotImplementedError

    def bulk_create(
        self, edges: typing.Sequence[EdgeType], *, verbose: bool | None = None
    ) -> typing.List[EdgeType]:
        raise NotImplementedError

    def list(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> Pagination[EdgeType]:
        raise NotImplementedError

    def gen(
        self,
        *,
        after: typing.Optional[typing.Text] = None,
        before: typing.Optional[typing.Text] = None,
        limit: int = 20,
        order: typing.Literal["asc", "desc"] = "asc",
        verbose: bool | None = None,
    ) -> typing.Generator[EdgeType, None, None]:
        raise NotImplementedError

    def count(self, *, verbose: bool | None = None) -> int:
        raise NotImplementedError
