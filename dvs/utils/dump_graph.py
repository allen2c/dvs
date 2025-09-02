import json
import pathlib
import typing

if typing.TYPE_CHECKING:
    import networkx as nx


def dump_graph(
    G: "nx.DiGraph",
    path: pathlib.Path | str,
    *,
    format: typing.Literal["node_link", "gexf"],
) -> pathlib.Path:
    import networkx as nx

    path = pathlib.Path(path)
    if format == "node_link":
        data = nx.node_link_data(G)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)

    elif format == "gexf":
        nx.write_gexf(G, path)

    else:
        raise ValueError(f"Unsupported format: {format}")

    return path
