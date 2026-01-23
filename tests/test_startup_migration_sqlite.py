"""Tests for startup database migration hook.

Tests cover:
- _run_startup_migrations() running automatically on server start
- Migration is idempotent and handles various initial states

See specs/custom-prompt-config.md § Migration for requirements.
"""

from unittest.mock import patch

import pytest
from sqlalchemy import text

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.migrations import SchemaVersion, detect_schema_version
from ragzoom.server.app import _run_startup_migrations


class TestStartupMigration:
    """Tests for _run_startup_migrations function."""

    def test_startup_migration_runs(self, sqlite_backend: SQLiteStorageBackend) -> None:
        """Verify migration runs automatically when called during startup.

        Requirements from spec:
        - On server start, migration runs automatically if needed
        - V1 schema (summary_system_prompt) is migrated to V2 (summarization_guidance)
        """
        # Verify starting state: V1 schema
        version = detect_schema_version(sqlite_backend.engine)
        assert version == SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT

        # Insert test data before migration
        with sqlite_backend.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO documents "
                    "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                    "VALUES ('test-doc', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                    "datetime('now'), 'custom guidance')"
                )
            )

        # Run startup migrations
        _run_startup_migrations(sqlite_backend)

        # Verify migration occurred
        version = detect_schema_version(sqlite_backend.engine)
        assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

        # Verify data was preserved
        with sqlite_backend.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT summarization_guidance FROM documents WHERE id = 'test-doc'"
                )
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "custom guidance"

    def test_startup_migration_idempotent(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Verify running startup migrations multiple times is safe."""
        # Run migrations twice
        _run_startup_migrations(sqlite_backend)
        _run_startup_migrations(sqlite_backend)

        # Should be V2 after both runs
        version = detect_schema_version(sqlite_backend.engine)
        assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

    def test_startup_migration_skips_v2_schema(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Verify migration is skipped when schema is already V2."""
        # First migrate to V2
        _run_startup_migrations(sqlite_backend)

        # Verify we're at V2
        version = detect_schema_version(sqlite_backend.engine)
        assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

        # Run again with mock to verify migrate_summary_prompt_column is not called
        with patch("ragzoom.migrations.migrate_summary_prompt_column") as mock_migrate:
            _run_startup_migrations(sqlite_backend)
            mock_migrate.assert_not_called()

    def test_startup_migration_handles_missing_table(self) -> None:
        """Verify startup migration handles case where documents table doesn't exist.

        This can happen on first startup before any documents are indexed.
        """
        # Create a backend with no documents table
        backend = SQLiteStorageBackend("sqlite:///:memory:")

        # Drop the documents table to simulate first startup
        with backend.engine.begin() as conn:
            conn.execute(text("DROP TABLE IF EXISTS documents"))

        # Should not raise - migration is skipped gracefully
        _run_startup_migrations(backend)

        backend.close()

    def test_startup_migration_logs_progress(
        self, sqlite_backend: SQLiteStorageBackend, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify migration logs appropriate messages."""
        import logging

        with caplog.at_level(logging.INFO):
            _run_startup_migrations(sqlite_backend)

        # Should log migration start and completion
        assert "Migrating database schema" in caplog.text
        assert "completed successfully" in caplog.text
