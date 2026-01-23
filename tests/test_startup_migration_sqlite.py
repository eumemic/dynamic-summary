"""Tests for startup database migration hook.

Tests cover:
- _run_startup_migrations() running automatically on server start
- Migration is idempotent and handles various initial states

See specs/custom-prompt-config.md § Migration for requirements.
"""

from typing import Protocol
from unittest.mock import patch

import pytest
from sqlalchemy import DateTime, Integer, String, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.migrations import SchemaVersion, detect_schema_version
from ragzoom.server.app import _run_startup_migrations


class StorageBackendProtocol(Protocol):
    """Protocol for storage backend with engine property."""

    @property
    def engine(self) -> Engine: ...


class V1StorageBackend:
    """A minimal storage backend with V1 schema for testing migrations."""

    def __init__(self) -> None:
        self._engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        class V1Base(DeclarativeBase):
            pass

        class V1Document(V1Base):
            __tablename__ = "documents"
            id: Mapped[str] = mapped_column(String, primary_key=True)
            user_id: Mapped[str | None] = mapped_column(String, nullable=True)
            file_path: Mapped[str | None] = mapped_column(String, nullable=True)
            indexed_at: Mapped[str] = mapped_column(DateTime, nullable=True)
            embedding_model: Mapped[str] = mapped_column(String, nullable=False)
            summary_model: Mapped[str] = mapped_column(String, nullable=False)
            is_temporal: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
            summary_system_prompt: Mapped[str | None] = mapped_column(
                String, nullable=True, default=None
            )

        V1Base.metadata.create_all(self._engine)

    @property
    def engine(self) -> Engine:
        return self._engine

    def close(self) -> None:
        self._engine.dispose()


class TestStartupMigration:
    """Tests for _run_startup_migrations function."""

    def test_startup_migration_runs(self) -> None:
        """Verify migration runs automatically when called during startup.

        Requirements from spec:
        - On server start, migration runs automatically if needed
        - V1 schema (summary_system_prompt) is migrated to V2 (summarization_guidance)
        """
        backend = V1StorageBackend()
        try:
            # Verify starting state: V1 schema
            version = detect_schema_version(backend.engine)
            assert version == SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT

            # Insert test data before migration
            with backend.engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                        "VALUES ('test-doc', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                        "datetime('now'), 'custom guidance')"
                    )
                )

            # Run startup migrations
            _run_startup_migrations(backend)  # type: ignore[arg-type]

            # Verify migration occurred
            version = detect_schema_version(backend.engine)
            assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

            # Verify data was preserved
            with backend.engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT summarization_guidance FROM documents WHERE id = 'test-doc'"
                    )
                )
                row = result.fetchone()
                assert row is not None
                assert row[0] == "custom guidance"
        finally:
            backend.close()

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
        # SQLiteStorageBackend creates V2 schema directly now
        # Verify we're at V2
        version = detect_schema_version(sqlite_backend.engine)
        assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

        # Run with mock to verify migrate_summary_prompt_column is not called
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
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Verify migration logs appropriate messages."""
        import logging

        backend = V1StorageBackend()
        try:
            with caplog.at_level(logging.INFO):
                _run_startup_migrations(backend)  # type: ignore[arg-type]

            # Should log migration start and completion
            assert "Migrating database schema" in caplog.text
            assert "completed successfully" in caplog.text
        finally:
            backend.close()
