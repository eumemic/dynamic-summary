"""Tests for PostgreSQL database schema migration functions.

Tests verify correct SQL is generated for PostgreSQL dialect.
Uses mock engine to avoid requiring a real PostgreSQL instance.
"""

from unittest.mock import MagicMock, patch

import pytest


class TestMigrateSummaryPromptColumnPostgres:
    """Tests for migrate_summary_prompt_column function with PostgreSQL."""

    def test_rename_column_postgres(self) -> None:
        """Verify migrate_summary_prompt_column generates correct PostgreSQL SQL.

        Requirements from spec:
        - Column is renamed from summary_system_prompt to summarization_guidance
        - Uses PostgreSQL DO $$ block for conditional execution
        - Migration is idempotent (checks if column exists before renaming)
        """
        from ragzoom.migrations import SchemaVersion, migrate_summary_prompt_column

        # Create mock engine that reports as PostgreSQL
        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"

        executed_sql: list[str] = []

        # Mock the connection context manager
        mock_conn = MagicMock()
        mock_conn.execute = lambda stmt: executed_sql.append(str(stmt))
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=None)

        # Mock detect_schema_version to return V1 (needs migration)
        with patch(
            "ragzoom.migrations.detect_schema_version",
            return_value=SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT,
        ):
            migrate_summary_prompt_column(mock_engine)

        # Verify SQL was executed
        assert len(executed_sql) == 1
        sql = executed_sql[0]

        # Verify it's a PostgreSQL DO block with conditional rename
        assert "DO $$" in sql
        assert "information_schema.columns" in sql
        assert "summary_system_prompt" in sql
        assert "summarization_guidance" in sql
        assert "ALTER TABLE documents" in sql
        assert "RENAME COLUMN" in sql

    def test_rename_column_postgres_idempotent(self) -> None:
        """Verify migration skips if already at V2 schema."""
        from ragzoom.migrations import SchemaVersion, migrate_summary_prompt_column

        mock_engine = MagicMock()
        mock_engine.dialect.name = "postgresql"

        executed_sql: list[str] = []
        mock_conn = MagicMock()
        mock_conn.execute = lambda stmt: executed_sql.append(str(stmt))
        mock_engine.begin.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_engine.begin.return_value.__exit__ = MagicMock(return_value=None)

        # Mock detect_schema_version to return V2 (already migrated)
        with patch(
            "ragzoom.migrations.detect_schema_version",
            return_value=SchemaVersion.V2_SUMMARIZATION_GUIDANCE,
        ):
            migrate_summary_prompt_column(mock_engine)

        # No SQL should be executed - migration should skip
        assert len(executed_sql) == 0

    def test_rename_column_postgres_unsupported_dialect(self) -> None:
        """Verify migration raises for unsupported dialects."""
        from ragzoom.migrations import SchemaVersion, migrate_summary_prompt_column

        mock_engine = MagicMock()
        mock_engine.dialect.name = "mysql"  # Unsupported

        with patch(
            "ragzoom.migrations.detect_schema_version",
            return_value=SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT,
        ):
            with pytest.raises(ValueError, match="Unsupported database dialect"):
                migrate_summary_prompt_column(mock_engine)
