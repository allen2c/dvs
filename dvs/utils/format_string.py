def format_string(s: str) -> str:
    return s.casefold().replace("-", "_").replace(" ", "_").strip()
