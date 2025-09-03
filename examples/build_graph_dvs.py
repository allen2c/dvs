import asyncio
import json
import pathlib
import typing

import agents
import diskcache
import logging_bullet_train as lbt
import openai
import openai_embeddings_model as oai_emb_model
from aps_agent import APSAgent
from ner_agent import NerAgent
from openai_embeddings_model import (
    AsyncOpenAIEmbeddingsModel,
)
from openai_embeddings_model import ModelSettings as EmbeddingModelSettings
from rich.console import Console

import dvs
from dvs.types.document import Document
from dvs.utils.build_graph_from_documents import build_graph_from_documents
from dvs.utils.dump_graph import dump_graph
from dvs.utils.load_documents_from_directory import load_documents_from_directory
from dvs.utils.loads_graph import load_graph

VERBOSE = True
MAX_CONCURRENCY = 4

lbt.set_logger("dvs")

console = Console()

data_root = pathlib.Path(__file__).parent.parent.joinpath("data")
documents_dir = data_root.joinpath("demo_documents")
duckdb_path = data_root.joinpath("example_graph_dvs.duckdb")
graph_stem_name = "example_graph_dvs"
label_embeddings_path = data_root.joinpath("label_embeddings.json")

if duckdb_path.is_file():
    duckdb_path.unlink()

dvs_settings = dvs.Settings(DUCKDB_PATH=str(duckdb_path))

dvs_client = dvs.DVS(
    dvs_settings,
    model=oai_emb_model.OpenAIEmbeddingsModel(
        model=oai_emb_model.EmbeddingModelType.TEXT_EMBEDDING_3_SMALL,
        openai_client=openai.OpenAI(),
        cache=diskcache.Cache("cache/dvs/embeddings.cache"),
    ),
    model_settings=oai_emb_model.ModelSettings(dimensions=512),
    verbose=VERBOSE,
)

openai_client = openai.AsyncOpenAI()
chat_model = agents.OpenAIResponsesModel(
    model="gpt-4.1-nano",
    openai_client=openai_client,
)
emb_model = AsyncOpenAIEmbeddingsModel("text-embedding-3-small", openai_client)
emb_settings = EmbeddingModelSettings(dimensions=512)

aps_agent = APSAgent()
ner_agent = NerAgent()


def load_documents(directory_path: pathlib.Path | str) -> list[Document]:
    documents: list[Document] = load_documents_from_directory(
        data_root.joinpath("demo_documents")
    )
    if not documents:
        raise ValueError(f"No documents found in {directory_path}")
    return documents


async def main():
    console.rule("[bold green]Graph DVS Construction Pipeline[/bold green]")

    # --- Step 1: Document Loading ---
    console.rule("[bold blue]Step 1: Document Loading[/bold blue]")
    documents = load_documents(documents_dir)

    # --- Step 2: Build DVS ---
    console.rule("[bold blue]Step 2: Build DVS[/bold blue]")
    created_result = dvs_client.add(documents, verbose=VERBOSE)
    console.log(f"Created DVS result: {created_result}")

    # --- Step 2.5: Optional Draw embeddings on 2D plane ---
    console.rule("[bold blue]Step 2.5: Draw embeddings on 2D plane[/bold blue]")
    points = [p for p in dvs_client.db.points.gen(limit=100, with_embedding=True)]
    label_embeddings: typing.List[typing.Tuple[str, typing.List[float]]] = [
        (dvs_client.db.documents.retrieve(p.document_id).name, p.to_python())
        for p in points
    ]
    label_embeddings_path.write_text(json.dumps(label_embeddings))

    # --- Step 3: Build Graph ---
    console.rule("[bold blue]Step 3: Build Graph[/bold blue]")
    graph = await build_graph_from_documents(
        documents,
        chat_model=chat_model,
        embeddings_model=emb_model,
        embeddings_model_settings=emb_settings,
        aps_agent=aps_agent,
        ner_agent=ner_agent,
        max_concurrency=MAX_CONCURRENCY,
        verbose=VERBOSE,
    )
    dump_graph(graph, data_root.joinpath(f"{graph_stem_name}.json"), format="node_link")
    graph_path = dump_graph(
        graph, data_root.joinpath(f"{graph_stem_name}.gexf"), format="gexf"
    )
    graph = load_graph(graph_path, format="gexf")
    console.log(f"Built Graph: {graph_path}")


if __name__ == "__main__":
    asyncio.run(main())
