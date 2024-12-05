import json
import typing


def ensure_dict(data: typing.Any) -> typing.Dict:
    if not data:
        return {}
    elif isinstance(data, typing.Text):
        return json.loads(data)

    return json.loads(json.dumps(data, default=str))
