# Example dataset: BBC News
# http://mlg.ucd.ie/datasets/bbc.html

import asyncio
from pathlib import Path

from rich.prompt import Confirm

from dvs import DVS
from dvs.config import console, settings
from dvs.utils.datasets import download_documents

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


client = DVS(debug=True)


async def main():
    docs = download_documents(name="bbc", overwrite=False)
    console.print(docs[0])

    client.add(docs)

    results = await client.search(query="What is the weather in London?", debug=True)
    console.print(results)


if __name__ == "__main__":
    asyncio.run(main())
