from typing import List, Text

from rich.pretty import pretty_repr

DISPLAY_SQL_QUERY = "=== Start of SQL ===\n{sql}\n=== End of SQL ==="
DISPLAY_SQL_PARAMS = "=== Start of SQL Params ===\n{params}\n=== End of SQL Params ==="


def display_sql_parameters(
    params: List, *, max_length: int = 64, max_lines: int = 5
) -> List[Text]:
    out: List[Text] = []
    for param in params[:max_lines]:
        out.append(pretty_repr(str(param), max_string=max_length))
    if len(params) > max_lines:
        out.append("...")
    return out
