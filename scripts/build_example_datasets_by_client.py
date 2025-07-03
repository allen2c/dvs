# Example dataset: BBC News
# http://mlg.ucd.ie/datasets/bbc.html

import asyncio

from rich.prompt import Confirm

from dvs import DVS
from dvs.config import Settings
from dvs.utils.datasets import download_documents


async def main():
    settings = Settings()
    console = settings.console

    if settings.duckdb_path.exists():
        is_rebuild = Confirm.ask(
            f"DuckDB database already exists: {settings.duckdb_path}. Rebuild?"
        )
        if is_rebuild:
            settings.duckdb_path.unlink()
            console.print(f"Deleted DuckDB database: {settings.duckdb_path}")
        else:
            console.print("Aborted")
            raise SystemExit()

    client = DVS(debug=True)

    docs = download_documents(name="bbc", overwrite=False)
    console.print(docs[0])

    client.add(docs)

    results = await client.search(query="What is the weather in London?", debug=True)
    console.print(results)


if __name__ == "__main__":
    asyncio.run(main())
