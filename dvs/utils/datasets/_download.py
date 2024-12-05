from typing import List, Optional, Text

from dvs.types.document import Document


def download_documents(
    name: Text = "bbc",
    *,
    overwrite: Optional[bool] = None,
) -> List[Document]:
    if name == "bbc":
        import dvs.utils.datasets.bbc

        return dvs.utils.datasets.bbc.download_documents(overwrite=overwrite)
    raise ValueError(f"Unknown dataset: {name}")
