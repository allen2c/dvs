import asyncio
import logging
import pathlib
import typing

import agents
import logging_bullet_train as lbt
import networkx as nx
import openai
from aps_agent import APSAgent
from ner_agent import NerAgent
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
)
from openai_embeddings_model import ModelSettings as EmbeddingModelSettings
from rich.console import Console

from dvs.types.document import Document
from dvs.utils.aps import get_facts
from dvs.utils.dump_graph import dump_graph
from dvs.utils.export_graph_png import export_graph_png
from dvs.utils.get_valid_canonical_map import get_valid_canonical_map
from dvs.utils.graph_from_triplets import graph_from_triplets
from dvs.utils.load_documents_from_directory import load_documents_from_directory
from dvs.utils.ner import extract_entities, extract_relations

T = typing.TypeVar("T")

lbt.set_logger("dvs", level=logging.DEBUG)

OPENAI_CONCURRENCY = 4


data_root = pathlib.Path(__file__).parent.parent.joinpath("data")
graph_stem_name = "knowledge_graph"

openai_client = openai.AsyncOpenAI()
chat_model = agents.OpenAIResponsesModel(
    model="gpt-5-nano",
    openai_client=openai_client,
)
emb_model = AsyncOpenAIEmbeddingsModel("text-embedding-3-small", openai_client)
emb_settings = EmbeddingModelSettings(dimensions=512)

aps_agent = APSAgent()
ner_agent = NerAgent()


async def main():
    console = Console()
    console.rule("[bold green]Knowledge Graph Construction Pipeline[/bold green]")

    # --- Step 1: Document Loading ---
    console.rule("[bold blue]Step 1: Document Loading[/bold blue]")
    documents: list[Document] = load_documents_from_directory("./data/demo_documents/")
    if not documents:
        return

    # --- Step 2: Abstractive Proposition Segmentation (APS) ---
    console.rule(
        "[bold blue]Step 2: Abstractive Proposition Segmentation (APS)[/bold blue]"
    )
    all_facts = await get_facts(documents, aps_agent=aps_agent, model=chat_model)

    # --- Step 3: Relation Extraction ---
    console.rule("[bold blue]Step 3: Relation Extraction[/bold blue]")
    all_triplets = await extract_relations(
        all_facts, ner_agent=ner_agent, model=chat_model
    )

    # --- Step 3.5: Optional NER Extraction ---
    console.rule(
        "[bold blue]Step 3.5: Optional Named Entity Recognition Extraction[/bold blue]"
    )
    extra_entities = await extract_entities(
        all_facts,
        ner_agent=ner_agent,
        model=chat_model,
        max_concurrency=OPENAI_CONCURRENCY,
    )

    # --- Step 4: Collect All Entities from Triplets and Extra Entities ---
    console.rule(
        "[bold blue]Step 4: Collect All Entities from Triplets and Extra Entities"
        + "[/bold blue]"
    )
    all_entities: typing.Set[str] = set()
    for triplet in all_triplets:
        all_entities.add(triplet.subject)
        all_entities.add(triplet.object)
    for entity in extra_entities:
        all_entities.add(entity.value)

    unique_entities = sorted(list(all_entities))
    console.log(f"✅ Collected {len(unique_entities)} unique entities from triplets.")

    # --- Step 5: Node Normalization Pipeline ---
    console.rule(
        "[bold blue]"
        + "Step 5: Node Normalization (Embedding -> Clustering -> Validation)"
        + "[/bold blue]"
    )
    canonical_map = await get_valid_canonical_map(
        unique_entities,
        embedding_model=emb_model,
        model_settings=emb_settings,
        ner_agent=ner_agent,
        chat_model=chat_model,
        max_concurrency=OPENAI_CONCURRENCY,
        verbose=False,
    )

    # --- Step 6: Build the Final Knowledge Graph ---
    console.rule("[bold blue]Step 6: Building Final Knowledge Graph[/bold blue]")
    G = graph_from_triplets(all_triplets, canonical_map=canonical_map)

    # --- Step 7: Export Graph ---
    graph_path = dump_graph(
        G, data_root.joinpath(f"{graph_stem_name}.json"), format="node_link"
    )
    console.log(f"\nGraph saved to [green]{graph_path}[/green].")
    export_graph_png(G, data_root.joinpath(f"{graph_stem_name}.png"))
    nx.write_gexf(G, data_root.joinpath(f"{graph_stem_name}.gexf"))


if __name__ == "__main__":
    asyncio.run(main())
