# examples/bbc_simple_demo.py

import logging
import pathlib

import diskcache
import logging_bullet_train as lbt
import openai
import openai_embeddings_model as oai_emb_model
import rich.console

import dvs
from dvs.utils.datasets import download_documents

DATASET_NAME = "bbc"
OVERWRITE = False
DUCKDB_PATH = pathlib.Path("./data/bbc.duckdb")
EMBEDDING_CACHE_PATH = pathlib.Path("./cache/dvs/embeddings.cache")
MODEL = "text-embedding-3-small"
DIMENSIONS = 512


console = rich.console.Console()
logger = logging.getLogger(__name__)

lbt.set_logger(logger, level=logging.DEBUG)
lbt.set_logger("dvs", level=logging.DEBUG)


def main():
    """Download datasets with specified parameters."""

    docs = download_documents(DATASET_NAME, overwrite=OVERWRITE)

    dvs_settings = dvs.Settings(DUCKDB_PATH=str(DUCKDB_PATH))
    dvs_client = dvs.DVS(
        dvs_settings,
        model=oai_emb_model.OpenAIEmbeddingsModel(
            model=MODEL,
            openai_client=openai.OpenAI(),
            cache=diskcache.Cache(EMBEDDING_CACHE_PATH),
        ),
        model_settings=oai_emb_model.ModelSettings(
            dimensions=DIMENSIONS,
        ),
        verbose=True,
    )

    console.print(f"Database file does not exist at {dvs_client.duckdb_path}")
    console.print(f"Adding {len(docs)} documents to the database")

    dvs_client.add(docs, verbose=True)

    console.print(f"Added {dvs_client.db.documents.count()} documents to the database")
    console.print(f"Added {dvs_client.db.points.count()} points to the database")


if __name__ == "__main__":
    main()
