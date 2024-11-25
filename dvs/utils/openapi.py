from typing import Dict, Optional, Sequence, Text

from dvs.utils.sql_stmts import SQL_STMT_CREATE_INDEX

COLUMN_DECLARATION_LINE = "{field_name} {sql_type} {not_null} {unique}"
DECLARATION_INDENT = "    "


# Mapping for types, adding JSON type
JSON_TO_SQL_TYPE_MAP = {
    "string": "TEXT",  # Default fallback for strings without max_length
    "text": "TEXT",  # We'll override for specific VARCHAR needs
    "integer": "INT",
    "boolean": "BOOLEAN",
    "number": "FLOAT",
    "object": "JSON",  # JSON type for metadata or other object fields
}


# Function to generate SQL CREATE TABLE statement with JSON support
def openapi_to_create_table_sql(
    schema: Dict,
    *,
    table_name: Text,
    primary_key: Optional[Text] = None,
    unique_fields: Optional[Sequence[Text]] = None,
    indexes: Optional[Sequence[Text]] = None,
) -> Text:
    unique_fields = unique_fields or []
    indexes = indexes or []

    columns = []
    index_statements = []

    for field_name, field_info in schema["properties"].items():
        field_type = field_info.get("type")

        # Handle array types
        if field_type == "array":
            # Detect the type of the array items (e.g., FLOAT[384])
            items_type = field_info["items"]["type"]
            array_size = field_info.get(
                "maxItems", 1
            )  # Default to size 1 if not provided

            # Mapping for the items type (e.g., "number" to "FLOAT")
            if items_type == "number":
                sql_type = f"FLOAT[{array_size}]"
            else:
                sql_type = f"TEXT[{array_size}]"  # Default to TEXT for other types

        # Handle VARCHAR for string types with maxLength
        elif "maxLength" in field_info and field_type == "string":
            sql_type = f"VARCHAR({field_info['maxLength']})"

        # Handle regular types (including JSON)
        else:
            sql_type = JSON_TO_SQL_TYPE_MAP.get(field_type, "TEXT")

        not_null = "NOT NULL" if field_name in schema.get("required", []) else ""
        unique = "UNIQUE" if field_name in unique_fields else ""
        columns.append(
            COLUMN_DECLARATION_LINE.format(
                field_name=field_name,
                sql_type=sql_type,
                not_null=not_null,
                unique=unique,
            ).strip()
        )

        # Add indexes
        if field_name in indexes:
            index_statements.append(
                SQL_STMT_CREATE_INDEX.format(
                    table_name=table_name, column_name=field_name
                ).strip()
            )

    # Add primary key
    if primary_key:
        columns.append(f"PRIMARY KEY ({primary_key})")

    columns_sql = f",\n{DECLARATION_INDENT}".join(columns)
    create_table_sql = (
        f"CREATE TABLE {table_name} (\n{DECLARATION_INDENT}{columns_sql}\n);"
    )

    # Combine CREATE TABLE and CREATE INDEX statements
    full_sql = create_table_sql + "\n" + ";\n".join(index_statements)
    return full_sql
