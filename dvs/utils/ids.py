from typing import Literal, Text

import uuid_utils as uuid


def get_id(type: Literal["point", "pt", "document", "doc"]) -> Text:
    if type in ("point", "pt"):
        return "pt-" + str(uuid.uuid4())
    elif type in ("document", "doc"):
        return "doc-" + str(uuid.uuid4())
    else:
        raise ValueError(f"Invalid type: {type}")
