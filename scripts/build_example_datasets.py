# Example dataset: BBC News
# http://mlg.ucd.ie/datasets/bbc.html

import asyncio
import zipfile
from pathlib import Path
from typing import Generator, List

import diskcache
import duckdb
import httpx
from openai import OpenAI
from rich.prompt import Confirm
from tqdm import tqdm

import dvs.utils.vss
from dvs.config import console, settings
from dvs.types.document import Document
from dvs.types.point import Point

DOWNLOAD_DIR = Path("./downloads").resolve()
TARGET_DIR = Path("./data").resolve()
TARGET_DB_PATH = Path(settings.DUCKDB_PATH).resolve()

if TARGET_DB_PATH.exists():
    if Confirm.ask(
        f"Target database path already exists: {TARGET_DB_PATH}. Overwrite?"
    ):
        TARGET_DB_PATH.unlink()
    else:
        console.print("Target database path already exists, exiting.")
        exit()


openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
cache = diskcache.Cache(settings.CACHE_PATH, size_limit=settings.CACHE_SIZE_LIMIT)


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

    points: List[Point] = [pt for doc in docs for pt in doc.to_points()]
    console.print(f"Created {len(points)} points")

    points = Point.set_embeddings_from_contents(
        points, docs, openai_client=openai_client, cache=cache, debug=True
    )
    console.print(f"Created {len(points)} embeddings")

    Document.objects.touch(
        conn=duckdb.connect(TARGET_DB_PATH), raise_if_exists=True, debug=True
    )
    Point.objects.touch(
        conn=duckdb.connect(TARGET_DB_PATH), raise_if_exists=True, debug=True
    )

    docs = Document.objects.bulk_create(
        docs, conn=duckdb.connect(TARGET_DB_PATH), debug=True
    )
    points = Point.objects.bulk_create(
        points, conn=duckdb.connect(TARGET_DB_PATH), debug=True
    )

    results = await dvs.utils.vss.vector_search(
        vector=openai_client.embeddings.create(
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
