# Example dataset: BBC News
# http://mlg.ucd.ie/datasets/bbc.html

import asyncio
import zipfile
from pathlib import Path

import httpx

DOWNLOAD_DIR = Path("./downloads").resolve()
TARGET_DIR = Path("./data").resolve()


async def download_bbc_news_dataset() -> Path:
    url = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        zip_path = DOWNLOAD_DIR / "bbc-fulltext.zip"
        zip_path.write_bytes(response.content)
        return zip_path


def unzip_bbc_news_dataset(zip_path: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(TARGET_DIR)
    return TARGET_DIR


async def main():
    zip_path = await download_bbc_news_dataset()
    print(f"Downloaded BBC News dataset to {zip_path}")
    unzip_bbc_news_dataset(zip_path)
    print(f"Unzipped BBC News dataset to {TARGET_DIR / 'bbc'}")


if __name__ == "__main__":
    asyncio.run(main())
