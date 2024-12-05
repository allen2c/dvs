import typing

from dvs.config import settings
from dvs.types.document import Document


def test_qs(documents: typing.List[Document]):
    print()
    print(settings.DUCKDB_PATH)
    print(len(documents))
