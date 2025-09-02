import asyncio
import pathlib

import agents
import diskcache
import logging_bullet_train as lbt
import openai
import openai_embeddings_model as oai_emb_model
from aps_agent import APSAgent
from ner_agent import NerAgent
from rich.console import Console

import dvs
from dvs.types.document import Document
from dvs.utils.load_documents_from_directory import load_documents_from_directory

VERBOSE = True

lbt.set_logger("dvs")

console = Console()

data_root = pathlib.Path(__file__).parent.parent.joinpath("data")
documents_dir = data_root.joinpath("demo_documents")
duckdb_path = data_root.joinpath("example_graph_dvs.duckdb")

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
    model="gpt-5-nano",
    openai_client=openai_client,
)

aps_agent = APSAgent()
ner_agent = NerAgent()


def load_documents(directory_path: pathlib.Path | str) -> list[Document]:
    documents: list[Document] = load_documents_from_directory("./data/demo_documents/")
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


if __name__ == "__main__":
    asyncio.run(main())
