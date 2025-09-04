import functools
import textwrap
import typing

import dvs
from dvs.utils.debug_print import debug_print
from dvs.utils.timer import Timer

if typing.TYPE_CHECKING:
    from dvs.db.graph.edges.api import Edges
    from dvs.db.graph.nodes.api import Nodes


class Graph:
    def __init__(self, dvs: dvs.DVS):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        self.nodes.touch(verbose=verbose)
        self.edges.touch(verbose=verbose)

        with Timer() as timer:
            create_table_sql = textwrap.dedent(
                f"""
                CREATE PROPERTY GRAPH {dvs.DVS_GRAPH_TABLE_NAME}
                VERTEX TABLES (
                    {dvs.DVS_NODES_TABLE_NAME}
                )
                EDGE TABLES (
                    {dvs.DVS_EDGES_TABLE_NAME}
                    SOURCE KEY (from_node) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                    DESTINATION KEY (to_node) REFERENCES {dvs.DVS_NODES_TABLE_NAME} (node_id)
                    LABEL relation
                )
                """  # noqa: E501
            )
            self.dvs.conn.execute(create_table_sql)

        debug_print(
            create_table_sql,
            title=f"Creating table: '{dvs.DVS_GRAPH_TABLE_NAME}' with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        return True

    @functools.cached_property
    def nodes(self) -> "Nodes":
        from dvs.db.graph.nodes.api import Nodes

        return Nodes(self.dvs)

    @functools.cached_property
    def edges(self) -> "Edges":
        from dvs.db.graph.edges.api import Edges

        return Edges(self.dvs)
