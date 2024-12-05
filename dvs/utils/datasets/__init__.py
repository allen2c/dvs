from typing import List, Text

from dvs.types.document import Document


def download_documents(
    name: Text = "bbc",
) -> List[Document]:
    if name == "bbc":
        import dvs.utils.datasets.bbc

        return dvs.utils.datasets.bbc.download_documents()
    raise ValueError(f"Unknown dataset: {name}")
