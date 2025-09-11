import asyncio
import logging
import typing

import agents
from aps_agent import APSAgent
from ner_agent import NerAgent
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
)
from openai_embeddings_model import ModelSettings as EmbeddingModelSettings
from openai_embeddings_model import (
    OpenAIEmbeddingsModel,
)

if typing.TYPE_CHECKING:
    import networkx as nx

    from dvs.types.document import Document
    from dvs.types.edge import Edge
    from dvs.types.entity import Entity
    from dvs.types.node import Node
    from dvs.types.triplet import Triplet

logger = logging.getLogger(__name__)


async def triplets_entities_from_document(
    document: "Document",
    *,
    chat_model: agents.OpenAIResponsesModel | agents.OpenAIChatCompletionsModel,
    extract_extra_entities: bool = True,
    document_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
    model_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
    verbose: bool = False,
) -> typing.Tuple[list["Triplet"], list["Entity"]]:
    from dvs.utils.aps import get_facts
    from dvs.utils.ner import extract_entities, extract_relations

    async with document_semaphore:
        ner_agent = NerAgent()
        aps_agent = APSAgent()

        # Abstractive Proposition Segmentation (APS)
        all_facts = await get_facts(
            [document],
            aps_agent=aps_agent,
            model=chat_model,
            model_semaphore=model_semaphore,
            verbose=verbose,
        )

        # Relation Extraction
        all_triplets = await extract_relations(
            all_facts,
            ner_agent=ner_agent,
            model=chat_model,
            model_semaphore=model_semaphore,
            verbose=verbose,
        )

        # Optional NER Extraction
        if extract_extra_entities:
            extra_entities = await extract_entities(
                all_facts,
                ner_agent=ner_agent,
                model=chat_model,
                model_semaphore=model_semaphore,
                verbose=verbose,
            )
        else:
            extra_entities = []

        return (all_triplets, extra_entities)


async def canonical_map_from_triplets_entities(
    triplets: list["Triplet"],
    entities: list["Entity"] | None = None,
    *,
    embeddings_model: OpenAIEmbeddingsModel | AsyncOpenAIEmbeddingsModel,
    embeddings_model_settings: EmbeddingModelSettings,
    chat_model: agents.OpenAIResponsesModel | agents.OpenAIChatCompletionsModel,
    model_semaphore: asyncio.Semaphore = asyncio.Semaphore(1),
    verbose: bool = False,
) -> typing.Dict[str, str]:
    from dvs.utils.get_valid_canonical_map import get_valid_canonical_map

    ner_agent = NerAgent()

    if entities is None:
        entities = []
    # Collect All Labels from Triplets and Extra Entities
    unique_labels = to_unique_entities(triplets, entities)
    logger.info(f"✅ Collected {len(unique_labels)} unique entities from triplets.")

    return await get_valid_canonical_map(
        unique_labels,
        embedding_model=embeddings_model,
        model_settings=embeddings_model_settings,
        ner_agent=ner_agent,
        chat_model=chat_model,
        model_semaphore=model_semaphore,
        verbose=verbose,
    )


async def graph_from_triplets_entities(
    triplets: list["Triplet"],
    entities: list["Entity"],
    *,
    canonical_map: typing.Dict[str, str],
) -> tuple["nx.DiGraph", list["Node"], list["Edge"]]:
    import networkx as nx

    from dvs.types.edge import Edge
    from dvs.types.node import Node

    # Build the Final Knowledge Graph
    G = nx.DiGraph()
    labels_nodes_map: dict[str, "Node"] = {}  # label -> Node
    labels_edges_map: dict[tuple[str, str, str], "Edge"] = (
        {}
    )  # from_node, to_node, relation

    # Add entities into graph
    for __entity in entities:
        __nodes, __edges = __entity.to_nodes_edges(
            canonical_map=canonical_map,
            labels_nodes_map=labels_nodes_map,
            labels_edges_map=labels_edges_map,
        )
        _ = [G.add_node(__node.node_id, **__node.model_dump()) for __node in __nodes]
        _ = [
            G.add_edge(__edge.from_node_id, __edge.to_node_id, **__edge.model_dump())
            for __edge in __edges
        ]

    # Add triplets into graph
    for __triplet in triplets:
        __nodes, __edges = __triplet.to_nodes_edges(
            canonical_map=canonical_map,
            labels_nodes_map=labels_nodes_map,
            labels_edges_map=labels_edges_map,
        )
        _ = [G.add_node(__node.node_id, **__node.model_dump()) for __node in __nodes]
        _ = [
            G.add_edge(__edge.from_node_id, __edge.to_node_id, **__edge.model_dump())
            for __edge in __edges
        ]

    if None in labels_nodes_map:
        raise ValueError("None in labels_nodes_map")
    if None in labels_edges_map:
        raise ValueError("None in labels_edges_map")

    nodes: list[Node] = [
        Node.model_validate(node_data) for _, node_data in G.nodes(data=True)
    ]
    edges: list[Edge] = [
        Edge.model_validate(edge_data) for _, _, edge_data in G.edges(data=True)
    ]

    return (G, nodes, edges)


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
