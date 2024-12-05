# Example dataset: BBC News
# http://mlg.ucd.ie/datasets/bbc.html

import asyncio
from pathlib import Path
from typing import List

from rich.prompt import Confirm

import dvs.utils.vss
from dvs.config import console, settings
from dvs.types.document import Document
from dvs.types.point import Point
from dvs.utils.datasets import download_documents

openai_client = settings.openai_client
cache = settings.cache


if Path(settings.DUCKDB_PATH).exists():
    is_rebuild = Confirm.ask(
        f"DuckDB database already exists: {settings.DUCKDB_PATH}. Rebuild?"
    )
    if is_rebuild:
        Path(settings.DUCKDB_PATH).unlink()
        console.print(f"Deleted DuckDB database: {settings.DUCKDB_PATH}")
    else:
        console.print("Aborted")
        raise SystemExit()


async def main():
    docs = download_documents(name="bbc", overwrite=False)
    console.print(docs[0])

    points: List[Point] = [pt for doc in docs for pt in doc.to_points()]
    console.print(f"Created {len(points)} points")

    points = Point.set_embeddings_from_contents(
        points, docs, openai_client=openai_client, cache=cache, debug=True
    )
    console.print(f"Created {len(points)} embeddings")

    Document.objects.touch(conn=settings.duckdb_conn, raise_if_exists=True, debug=True)
    Point.objects.touch(conn=settings.duckdb_conn, raise_if_exists=True, debug=True)

    docs = Document.objects.bulk_create(docs, conn=settings.duckdb_conn, debug=True)

    points = Point.objects.bulk_create(points, conn=settings.duckdb_conn, debug=True)

    results = await dvs.utils.vss.vector_search(
        vector=openai_client.embeddings.create(
            input="What is the weather in London?",
            model=settings.EMBEDDING_MODEL,
            dimensions=settings.EMBEDDING_DIMENSIONS,
        )
        .data[0]
        .embedding,
        top_k=3,
        conn=settings.duckdb_conn,
        with_embedding=False,
    )
    console.print(results)


if __name__ == "__main__":
    asyncio.run(main())
