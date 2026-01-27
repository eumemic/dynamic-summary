"""Tests for PostgreSQL _run_migrations() in DatabaseManager.

Tests verify that the PostgreSQL DatabaseManager._run_migrations() method
includes the summary_system_prompt → summarization_guidance column rename
migration that exists in SQLite backend.

See Issue #4: PostgreSQL Schema Migration Missing
"""

import inspect


class TestPostgresRunMigrations:
    """Tests for DatabaseManager._run_migrations PostgreSQL migration coverage."""

    def test_postgres_run_migrations_renames_summary_column(self) -> None:
        """PostgreSQL _run_migrations should rename summary_system_prompt to summarization_guidance.

        This migration exists in:
        - ragzoom/migrations.py (migrate_summary_prompt_column)
        - ragzoom/backends/sqlite_db.py (inline in SqliteDatabaseManager.__post_init__)
        - ragzoom/server/app.py (_run_startup_migrations)

        But was missing from:
        - ragzoom/storage/database_manager.py (DatabaseManager._run_migrations)

        This test verifies the migration SQL is present in _run_migrations.
        """
        from ragzoom.storage.database_manager import DatabaseManager

        source = inspect.getsource(DatabaseManager._run_migrations)

        # Verify the migration SQL includes both column names and RENAME COLUMN
        assert (
            "summary_system_prompt" in source
        ), "PostgreSQL _run_migrations should reference summary_system_prompt column"
        assert (
            "summarization_guidance" in source
        ), "PostgreSQL _run_migrations should reference summarization_guidance column"
        assert (
            "RENAME COLUMN" in source
        ), "PostgreSQL _run_migrations should include RENAME COLUMN SQL"
