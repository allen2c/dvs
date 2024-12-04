# Example dataset: BBC News
# http://mlg.ucd.ie/datasets/bbc.html

import asyncio
import zipfile
from pathlib import Path
from typing import Generator

import duckdb
import httpx
from rich.prompt import Confirm
from tqdm import tqdm

import dvs.utils.vss
from dvs import DVS
from dvs.config import console, settings
from dvs.types.document import Document

DOWNLOAD_DIR = Path("./downloads").resolve()
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DIR = Path("./data").resolve()
TARGET_DIR.mkdir(parents=True, exist_ok=True)
TARGET_DB_PATH = Path(settings.DUCKDB_PATH).resolve()

if TARGET_DB_PATH.exists():
    if Confirm.ask(
        f"Target database path already exists: {TARGET_DB_PATH}. Overwrite?"
    ):
        TARGET_DB_PATH.unlink()
    else:
        console.print("Target database path already exists, exiting.")
        exit()


client = DVS(debug=True)


async def download_bbc_news_dataset() -> Path:
    download_filepath = DOWNLOAD_DIR / "bbc-fulltext.zip"
    if download_filepath.exists():
        return download_filepath
    url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        download_filepath.write_bytes(response.content)
        return download_filepath


def unzip_bbc_news_dataset(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    return TARGET_DIR


def walk_bbc_news_dataset(root_dir: Path) -> Generator[Path, None, None]:
    for path in root_dir.glob("**/*"):
        if path.is_file():
            yield path


def parse_bbc_news_document(filepath: Path) -> Document:
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read().strip()
    return Document.model_validate(
        {
            "name": filepath.name,
            "content": content,
            "content_md5": Document.hash_content(content),
            "metadata": {"file": filepath.name, "content_length": len(content)},
            "created_at": int(filepath.stat().st_ctime),
            "updated_at": int(filepath.stat().st_mtime),
        }
    )


async def main():
    zip_path = await download_bbc_news_dataset()
    console.print(f"Downloaded BBC News dataset to {zip_path}")
    unzip_bbc_news_dataset(zip_path)
    console.print(f"Unzipped BBC News dataset to {TARGET_DIR / 'bbc'}")
    docs = [
        parse_bbc_news_document(path)
        for path in tqdm(
            walk_bbc_news_dataset(TARGET_DIR / "bbc"), desc="Parsing documents"
        )
    ]
    console.print(f"Parsed {len(docs)} documents")
    console.print(docs[0])

    client.add(docs)

    results = await dvs.utils.vss.vector_search(
        vector=client.openai_client.embeddings.create(
            input="What is the weather in London?",
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        .data[0]
        .embedding,
        top_k=3,
        conn=duckdb.connect(TARGET_DB_PATH),
        with_embedding=False,
    )
    console.print(results)


if __name__ == "__main__":
    asyncio.run(main())
