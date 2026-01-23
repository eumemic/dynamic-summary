"""Database schema migration utilities.

This module provides functions for detecting and migrating database schema versions,
particularly for the summary_system_prompt → summarization_guidance field rename.

See specs/custom-prompt-config.md § Migration for requirements.
"""

from enum import Enum

from sqlalchemy import text
from sqlalchemy.engine import Engine


class SchemaVersion(Enum):
    """Database schema versions for the Document table prompt column."""

    V1_SUMMARY_SYSTEM_PROMPT = "v1_summary_system_prompt"
    V2_SUMMARIZATION_GUIDANCE = "v2_summarization_guidance"


def detect_schema_version(engine: Engine) -> SchemaVersion:
    """Detect which schema version the database uses for the prompt column.

    Queries the database metadata to determine if the Document table uses
    the old `summary_system_prompt` column or the new `summarization_guidance` column.

    Args:
        engine: SQLAlchemy engine connected to the database

    Returns:
        SchemaVersion indicating which column exists

    Raises:
        ValueError: If the documents table doesn't exist or neither column is found

    Note:
        If both columns exist (migration in progress), returns V2 since the
        new column takes precedence.
    """
    dialect = engine.dialect.name

    with engine.connect() as conn:
        if dialect == "sqlite":
            # Check if documents table exists
            result = conn.execute(
                text(
                    "SELECT name FROM sqlite_master "
                    "WHERE type='table' AND name='documents'"
                )
            )
            if result.fetchone() is None:
                raise ValueError("documents table not found in database")

            # Use PRAGMA table_info to get column names
            result = conn.execute(text("PRAGMA table_info(documents)"))
            columns = {row[1] for row in result.fetchall()}  # row[1] is column name

        elif dialect == "postgresql":
            # Check if documents table exists
            result = conn.execute(
                text(
                    "SELECT EXISTS ("
                    "  SELECT 1 FROM information_schema.tables "
                    "  WHERE table_name = 'documents'"
                    ")"
                )
            )
            if not result.scalar():
                raise ValueError("documents table not found in database")

            # Query information_schema for column names
            result = conn.execute(
                text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'documents'"
                )
            )
            columns = {row[0] for row in result.fetchall()}

        else:
            raise ValueError(f"Unsupported database dialect: {dialect}")

    # Check for columns - new column takes precedence if both exist
    has_new = "summarization_guidance" in columns
    has_old = "summary_system_prompt" in columns

    if has_new:
        return SchemaVersion.V2_SUMMARIZATION_GUIDANCE
    elif has_old:
        return SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT
    else:
        raise ValueError(
            "Invalid schema: neither summary_system_prompt nor "
            "summarization_guidance column found in documents table"
        )
