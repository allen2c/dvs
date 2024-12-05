import zipfile
from pathlib import Path
from typing import Generator, List, Text

import requests
from tqdm import tqdm

from dvs.config import console, settings
from dvs.types.document import Document

URL_BBC_NEWS_DATASET = "http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip"


def download_documents(
    *,
    url: Text = URL_BBC_NEWS_DATASET,
    download_dirpath: Text | Path = settings._temp_dir,
    target_dirpath: Text | Path = settings.APP_DATA_DIR,
) -> List[Document]:
    download_dirpath = Path(download_dirpath).resolve()
    download_dirpath.mkdir(parents=True, exist_ok=True)
    target_dirpath = Path(target_dirpath).resolve()
    target_dirpath.mkdir(parents=True, exist_ok=True)

    zip_path = download_bbc_news_dataset(url=url, download_dirpath=download_dirpath)
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
    url: Text = URL_BBC_NEWS_DATASET, download_dirpath: Text | Path = settings._temp_dir
) -> Path:
    download_filepath = Path(download_dirpath) / "bbc-fulltext.zip"
    if download_filepath.exists():
        return download_filepath

    response = requests.get(url)
    response.raise_for_status()
    download_filepath.write_bytes(response.content)
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
