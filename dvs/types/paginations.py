from typing import Generic, List, Optional, Text, TypeVar

from openai._base_client import BasePage, BaseSyncPage
from openai.pagination import SyncPage
from pydantic import Field

_T = TypeVar("_T")


class Pagination(SyncPage[_T], BaseSyncPage[_T], BasePage[_T], Generic[_T]):
    data: List[_T]
    object: Text
    first_id: Optional[Text] = Field(default=None)
    last_id: Optional[Text] = Field(default=None)
    has_more: bool = Field(default=False)
