import logging
import typing

import agents
from aps_agent import APSAgent
from ner_agent import NerAgent
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
)
from openai_embeddings_model import ModelSettings as EmbeddingModelSettings

if typing.TYPE_CHECKING:
    import networkx as nx

    from dvs.types.document import Document
    from dvs.types.entity import Entity
    from dvs.types.triplet import Triplet

logger = logging.getLogger(__name__)


async def build_graph_from_documents(
    documents: list["Document"],
    *,
    chat_model: agents.OpenAIResponsesModel | agents.OpenAIChatCompletionsModel,
    embeddings_model: AsyncOpenAIEmbeddingsModel,
    embeddings_model_settings: EmbeddingModelSettings,
    aps_agent: APSAgent,
    ner_agent: NerAgent,
    max_concurrency: int = 1,
    verbose: bool = False,
) -> "nx.DiGraph":
    import networkx as nx

    from dvs.types.edge import Edge
    from dvs.types.node import Node
    from dvs.utils.aps import get_facts
    from dvs.utils.get_valid_canonical_map import get_valid_canonical_map
    from dvs.utils.ner import extract_entities, extract_relations

    if not documents:
        raise ValueError("No documents provided")

    # Abstractive Proposition Segmentation (APS)
    all_facts = await get_facts(
        documents,
        aps_agent=aps_agent,
        model=chat_model,
        max_concurrency=max_concurrency,
    )

    # Relation Extraction
    all_triplets = await extract_relations(
        all_facts,
        ner_agent=ner_agent,
        model=chat_model,
        max_concurrency=max_concurrency,
    )

    # Optional NER Extraction
    extra_entities = await extract_entities(
        all_facts,
        ner_agent=ner_agent,
        model=chat_model,
        max_concurrency=max_concurrency,
    )

    # Collect All Labels from Triplets and Extra Entities
    unique_labels = to_unique_entities(all_triplets, extra_entities)
    logger.info(f"✅ Collected {len(unique_labels)} unique entities from triplets.")

    # Node normalization canonical label names
    canonical_map = await get_valid_canonical_map(
        unique_labels,
        embedding_model=embeddings_model,
        model_settings=embeddings_model_settings,
        ner_agent=ner_agent,
        chat_model=chat_model,
        max_concurrency=max_concurrency,
        verbose=verbose,
    )

    # Build the Final Knowledge Graph
    G = nx.DiGraph()
    labels_nodes_map: dict[str, "Node"] = {}  # label -> Node
    labels_edges_map: dict[tuple[str, str, str], "Edge"] = (
        {}
    )  # from_node, to_node, relation

    # Add entities into graph
    for _entity in extra_entities:
        _nodes, _edges = _entity.to_nodes_edges(
            canonical_map=canonical_map,
            labels_nodes_map=labels_nodes_map,
            labels_edges_map=labels_edges_map,
        )
        for _node in _nodes:
            G.add_node(_node.node_id, **_node.model_dump())
        for _edge in _edges:
            G.add_edge(_edge.from_node_id, _edge.to_node_id, **_edge.model_dump())

    # Add triplets into graph
    for _triplet in all_triplets:
        _nodes, _edges = _triplet.to_nodes_edges(
            canonical_map=canonical_map,
            labels_nodes_map=labels_nodes_map,
            labels_edges_map=labels_edges_map,
        )
        for _node in _nodes:
            G.add_node(_node.node_id, **_node.model_dump())
        for _edge in _edges:
            G.add_edge(_edge.from_node_id, _edge.to_node_id, **_edge.model_dump())

    if None in labels_nodes_map:
        raise ValueError("None in labels_nodes_map")
    if None in labels_edges_map:
        raise ValueError("None in labels_edges_map")

    logger.info("✨ Knowledge Graph Construction Complete! ✨")
    logger.info(f"Total Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    return G


def to_unique_entities(
    triplets: list["Triplet"], extra_entities: list["Entity"]
) -> list[str]:
    all_entities: typing.Set[str] = set()
    for triplet in triplets:
        all_entities.add(triplet.subject)
        all_entities.add(triplet.object)
    for entity in extra_entities:
        all_entities.add(entity.value)
    return sorted(list(all_entities))
