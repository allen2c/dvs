import functools
import typing

import dvs

if typing.TYPE_CHECKING:
    from dvs.db.graph.edges.api import Edges
    from dvs.db.graph.nodes.api import Nodes


class Graph:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        raise NotImplementedError

    @functools.cached_property
    def nodes(self) -> Nodes:
        from dvs.db.graph.nodes.api import Nodes

        return Nodes(self.dvs)

    @functools.cached_property
    def edges(self) -> Edges:
        from dvs.db.graph.edges.api import Edges

        return Edges(self.dvs)
