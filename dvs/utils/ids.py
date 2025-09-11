from typing import Literal, Text

import uuid_utils as uuid


def get_id(
    type: Literal["point", "pt", "document", "doc", "node", "n", "edge", "e"],
) -> Text:
    """Generate unique ID with type prefix using UUID7."""
    if type in ("point", "pt"):
        return "pt-" + str(uuid.uuid7())
    elif type in ("document", "doc"):
        return "doc-" + str(uuid.uuid7())
    elif type in ("node", "n"):
        return "n-" + str(uuid.uuid7())
    elif type in ("edge", "e"):
        return "e-" + str(uuid.uuid7())
    else:
        raise ValueError(f"Invalid type: {type}")
