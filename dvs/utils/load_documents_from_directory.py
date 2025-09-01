import logging
import pathlib
import typing

if typing.TYPE_CHECKING:
    from dvs.types.document import Document


logger = logging.getLogger(__name__)


def load_documents_from_directory(
    directory_path: pathlib.Path | str,
) -> typing.List["Document"]:
    from dvs.types.document import Document

    documents: list[Document] = []
    path = pathlib.Path(directory_path)
    if not path.exists() or not path.is_dir():
        logger.error(f"Error: Directory not found at {path.resolve()}")
        return documents
    for file_path in path.glob("*.txt"):
        try:
            content: str = file_path.read_text(encoding="utf-8")
            doc_name: str = file_path.stem
            documents.append(Document.from_content(content, name=doc_name))
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")

    logger.info(f"✅ Loaded {len(documents)} documents.")
    return documents
