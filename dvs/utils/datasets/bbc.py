import zipfile
from pathlib import Path
from typing import Generator, List, Optional, Text

import requests
from rich.prompt import Confirm
from tqdm import tqdm

from dvs.config import console, settings
from dvs.types.document import Document

URL_BBC_NEWS_DATASET = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"


def download_documents(
    *,
    url: Text = URL_BBC_NEWS_DATASET,
    download_dirpath: Text | Path = settings._temp_dir,
    target_dirpath: Text | Path = settings.APP_DATA_DIR,
    overwrite: Optional[bool] = None,
) -> List[Document]:
    download_dirpath = Path(download_dirpath).resolve()
    download_dirpath.mkdir(parents=True, exist_ok=True)
    target_dirpath = Path(target_dirpath).resolve()
    target_dirpath.mkdir(parents=True, exist_ok=True)

    zip_path = download_bbc_news_dataset(
        url=url, download_dirpath=download_dirpath, overwrite=overwrite
    )
    unzip_bbc_news_dataset(zip_path, target_dirpath=target_dirpath)
    docs = [
        parse_bbc_news_document(path)
        for path in tqdm(
            walk_bbc_news_dataset(target_dirpath / "bbc"), desc="Parsing documents"
        )
    ]
    console.print(f"Downloaded {len(docs)} documents")

    return docs


def download_bbc_news_dataset(
    url: Text = URL_BBC_NEWS_DATASET,
    download_dirpath: Text | Path = settings._temp_dir,
    overwrite: Optional[bool] = None,
) -> Path:
    download_filepath = Path(download_dirpath) / "bbc-fulltext.zip"
    if download_filepath.exists():
        if overwrite is None:
            overwrite = Confirm.ask(
                f"File already exists: {download_filepath}. Overwrite?"
            )
        if not overwrite:
            return download_filepath

    # Stream the download with progress bar
    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Get total file size from headers
    total_size = int(response.headers.get("content-length", 0))

    # Setup progress bar
    progress = tqdm(
        total=total_size, unit="iB", unit_scale=True, desc="Downloading BBC dataset"
    )

    # Download with chunks and update progress
    with open(download_filepath, "wb") as file:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            progress.update(size)
    progress.close()

    return download_filepath


def unzip_bbc_news_dataset(
    zip_path: Path, target_dirpath: Text | Path = settings.APP_DATA_DIR
) -> Path:
    target_dirpath = Path(target_dirpath).resolve()
    target_dirpath.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(target_dirpath)
    return target_dirpath


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
