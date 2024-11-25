from typing import List, Text

DISPLAY_SQL_QUERY = "=== Start of SQL ===\n{sql}\n=== End of SQL ==="
DISPLAY_SQL_PARAMS = "=== Start of SQL Params ===\n{params}\n=== End of SQL Params ==="


def display_sql_parameters(
    params: List, *, max_length: int = 128, max_lines: int = 10
) -> List[Text]:
    out: List[Text] = []
    for param in params[:max_lines]:
        param_str = str(param)
        if len(param_str) > max_length:
            param_str = param_str[: max_length - 3] + "..."
            out.append(param_str)
        else:
            out.append(param)
    if len(params) > max_lines:
        out.append("...")
    return out
