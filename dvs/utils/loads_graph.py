import json
import pathlib
import typing

if typing.TYPE_CHECKING:
    import networkx as nx


def load_graph(
    path: pathlib.Path | str, *, format: typing.Literal["node_link", "gexf"]
) -> "nx.DiGraph":
    import networkx as nx

    path = pathlib.Path(path)

    if format == "node_link":
        with open(path, "r") as f:
            data = json.load(f)
        return nx.node_link_graph(data)
    elif format == "gexf":
        return nx.read_gexf(path)
    else:
        raise ValueError(f"Unsupported format: {format}")
