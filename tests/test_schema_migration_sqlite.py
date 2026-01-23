"""Tests for database schema migration functions.

Tests cover:
- detect_schema_version() identifying column names in Document table
- Migration from summary_system_prompt to summarization_guidance (future phases)
"""

import pytest
from sqlalchemy import Integer, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from ragzoom.backends.sqlite_backend import SQLiteStorageBackend
from ragzoom.migrations import SchemaVersion, detect_schema_version


class TestDetectSchemaVersion:
    """Tests for detect_schema_version function."""

    def test_detect_schema_version_with_old_column(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Verify detect_schema_version returns V1 when summary_system_prompt exists."""
        # SQLite backend creates tables with summary_system_prompt column (current schema)
        version = detect_schema_version(sqlite_backend.db.engine)

        assert version == SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT

    def test_detect_schema_version_with_new_column(self) -> None:
        """Verify detect_schema_version returns V2 when summarization_guidance exists."""
        # Create a fresh backend and manually rename the column to simulate migrated schema
        backend = SQLiteStorageBackend("sqlite:///:memory:")
        try:
            # SQLite doesn't support direct column rename easily, so we need to
            # recreate the table. For test purposes, we'll add the new column
            # and drop the old one.
            with backend.db.engine.begin() as conn:
                # Add new column
                conn.execute(
                    text("ALTER TABLE documents ADD COLUMN summarization_guidance TEXT")
                )
                # SQLite doesn't support DROP COLUMN in older versions, but
                # for detection we just need the column to exist
                # We'll test that having BOTH columns still detects as V2 (new takes precedence)

            version = detect_schema_version(backend.db.engine)
            # When both columns exist, we treat it as V2 (migration in progress)
            assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE
        finally:
            backend.close()

    def test_detect_schema_version_with_neither_column(self) -> None:
        """Verify detect_schema_version raises when neither column exists."""
        # Create engine with custom schema missing the prompt column
        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        class TestBase(DeclarativeBase):
            pass

        class TestDocument(TestBase):
            __tablename__ = "documents"
            id: Mapped[str] = mapped_column(String, primary_key=True)
            embedding_model: Mapped[str] = mapped_column(String, nullable=False)
            summary_model: Mapped[str] = mapped_column(String, nullable=False)
            is_temporal: Mapped[int] = mapped_column(Integer, default=0)
            # Note: no summary_system_prompt or summarization_guidance column

        TestBase.metadata.create_all(engine)

        with pytest.raises(ValueError, match="neither.*column found"):
            detect_schema_version(engine)

        engine.dispose()

    def test_detect_schema_version_with_no_documents_table(self) -> None:
        """Verify detect_schema_version raises when documents table doesn't exist."""

        engine = create_engine(
            "sqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )

        with pytest.raises(ValueError, match="documents table not found"):
            detect_schema_version(engine)

        engine.dispose()
