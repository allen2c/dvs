import logging
import typing

if typing.TYPE_CHECKING:
    import networkx as nx
    from ner_agent import Triplet

logger = logging.getLogger(__name__)


def graph_from_triplets(
    triplets: list["Triplet"], *, canonical_map: dict[str, str] | None = None
) -> "nx.DiGraph":
    import networkx as nx

    G = nx.DiGraph()

    canonical_map = canonical_map or {}

    for canonical_name in canonical_map.values():
        G.add_node(canonical_name)

    for triplet in triplets:
        # Normalize subject and object using the complete map
        subject_norm = canonical_map.get(triplet.subject, triplet.subject)
        object_norm = canonical_map.get(triplet.object, triplet.object)

        G.add_node(subject_norm)
        G.add_node(object_norm)
        G.add_edge(subject_norm, object_norm, label=triplet.relation)

    logger.info("✨ Knowledge Graph Construction Complete! ✨")
    logger.info(f"Total Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    return G
