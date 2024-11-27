from textwrap import dedent
from typing import Final, Text

# SQL statement for vector search
SQL_STMT_VSS: Final[Text] = dedent(
    """
    WITH vector_search AS (
        SELECT {column_names_expr}, array_cosine_similarity(embedding, ?::FLOAT[{embedding_dimensions}]) AS relevance_score
        FROM {points_table_name}
        ORDER BY relevance_score DESC
        LIMIT {top_k}
    )
    SELECT p.*, d.*, p.metadata AS point_metadata, d.metadata AS document_metadata
    FROM vector_search p
    JOIN {documents_table_name} d ON p.document_id = d.document_id
    ORDER BY p.relevance_score DESC
    """  # noqa: E501
).strip()

# SQL Show tables
SQL_STMT_SHOW_TABLES: Final[Text] = "SHOW TABLES"

# SQL Drop table
SQL_STMT_DROP_TABLE: Final[Text] = "DROP TABLE IF EXISTS {{ table_name }} CASCADE"

# SQL Remove outdated points
SQL_STMT_REMOVE_OUTDATED_POINTS: Final[Text] = dedent(
    """
    DELETE FROM {{ table_name }} WHERE document_id = ? AND content_md5 != ?
    """
).strip()

# SQL Get by doc ids
SQL_STMT_GET_BY_DOC_IDS: Final[Text] = dedent(
    """
    SELECT {{ columns_expr }} FROM {{ table_name }} WHERE document_id IN ( {{ placeholders }} )
    """  # noqa: E501
).strip()

# SQL Create index
SQL_STMT_CREATE_INDEX: Final[Text] = (
    "CREATE INDEX idx_{table_name}_{column_name} ON {table_name} ({column_name});"
)

# SQL Create embedding index
SQL_STMT_CREATE_EMBEDDING_INDEX: Final[Text] = (
    "CREATE INDEX idx_{table_name}_{column_name} ON {table_name} "
    + "USING HNSW({column_name}) "
    + "WITH (metric = '{metric}');"  # You can choose 'l2sq' or 'ip' instead of 'cosine' if needed  # noqa: E501
)

# SQL Recreate embedding index
SQL_STMT_RECREATE_EMBEDDING_INDEX: Final[Text] = dedent(
    """
    DROP INDEX index_name;
    CREATE INDEX idx_{table_name}_{column_name} ON table_name USING HNSW (vector_column) WITH (metric = '{metric}');
    """  # noqa: E501
).strip()

# SQL HNSW compact index
SQL_STMT_HNSW_COMPACT_INDEX: Final[Text] = (
    "PRAGMA hnsw_compact_index('idx_{table_name}_{column_name}');"
)

# SQL Install extensions
SQL_STMT_INSTALL_EXTENSIONS: Final[Text] = dedent(
    """
    INSTALL vss;
    LOAD vss;
    INSTALL json;
    LOAD json;
    """
).strip()

# SQL Set HNSW experimental persistence
SQL_STMT_SET_HNSW_EXPERIMENTAL_PERSISTENCE: Final[Text] = (
    "SET hnsw_enable_experimental_persistence = true;"
)
