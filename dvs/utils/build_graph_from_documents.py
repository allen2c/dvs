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

    from dvs.utils.aps import get_facts
    from dvs.utils.get_valid_canonical_map import get_valid_canonical_map
    from dvs.utils.graph_from_triplets import graph_from_triplets
    from dvs.utils.ner import extract_entities, extract_relations

    if not documents:
        raise ValueError("No documents provided")

    # Abstractive Proposition Segmentation (APS)
    all_facts = await get_facts(documents, aps_agent=aps_agent, model=chat_model)

    # Relation Extraction
    all_triplets = await extract_relations(
        all_facts, ner_agent=ner_agent, model=chat_model
    )

    # Optional NER Extraction
    extra_entities = await extract_entities(
        all_facts,
        ner_agent=ner_agent,
        model=chat_model,
        max_concurrency=max_concurrency,
    )

    # Collect All Entities from Triplets and Extra Entities
    all_entities: typing.Set[str] = set()
    for triplet in all_triplets:
        all_entities.add(triplet.subject)
        all_entities.add(triplet.object)
    for entity in extra_entities:
        all_entities.add(entity.value)

    unique_entities = sorted(list(all_entities))
    logger.info(f"✅ Collected {len(unique_entities)} unique entities from triplets.")

    # Node Normalization Pipeline
    canonical_map = await get_valid_canonical_map(
        unique_entities,
        embedding_model=embeddings_model,
        model_settings=embeddings_model_settings,
        ner_agent=ner_agent,
        chat_model=chat_model,
        max_concurrency=max_concurrency,
        verbose=verbose,
    )

    # Build the Final Knowledge Graph
    G = graph_from_triplets(all_triplets, canonical_map=canonical_map)

    return G
