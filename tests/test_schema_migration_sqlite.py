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

    def test_detect_schema_version_with_only_new_column(self) -> None:
        """Verify detect_schema_version returns V2 when only summarization_guidance exists.

        This tests the final post-migration state where the old column no longer exists.
        """
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
            summarization_guidance: Mapped[str | None] = mapped_column(
                String, nullable=True
            )

        TestBase.metadata.create_all(engine)

        version = detect_schema_version(engine)
        assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

        engine.dispose()

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


class TestMigrateSummaryPromptColumn:
    """Tests for migrate_summary_prompt_column function."""

    def test_rename_column_sqlite(self, sqlite_backend: SQLiteStorageBackend) -> None:
        """Verify migrate_summary_prompt_column renames column in SQLite.

        Requirements from spec:
        - Column is renamed from summary_system_prompt to summarization_guidance
        - Existing data is preserved
        - Migration is idempotent (safe to run multiple times)
        """
        from ragzoom.migrations import migrate_summary_prompt_column

        # Verify starting state: V1 schema
        version = detect_schema_version(sqlite_backend.db.engine)
        assert version == SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT

        # Insert test data before migration
        with sqlite_backend.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO documents "
                    "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                    "VALUES ('test-doc-1', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                    "datetime('now'), 'test guidance')"
                )
            )

        # Run migration
        migrate_summary_prompt_column(sqlite_backend.db.engine)

        # Verify column was renamed
        version = detect_schema_version(sqlite_backend.db.engine)
        assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

        # Verify data was preserved in new column
        with sqlite_backend.db.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT summarization_guidance FROM documents WHERE id = 'test-doc-1'"
                )
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "test guidance"

    def test_rename_column_sqlite_idempotent(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Verify migration is idempotent - safe to run multiple times."""
        from ragzoom.migrations import migrate_summary_prompt_column

        # Insert test data
        with sqlite_backend.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO documents "
                    "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                    "VALUES ('test-doc-2', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                    "datetime('now'), 'some guidance')"
                )
            )

        # Run migration twice
        migrate_summary_prompt_column(sqlite_backend.db.engine)
        migrate_summary_prompt_column(sqlite_backend.db.engine)

        # Should still be V2 and data intact
        version = detect_schema_version(sqlite_backend.db.engine)
        assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

        with sqlite_backend.db.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT summarization_guidance FROM documents WHERE id = 'test-doc-2'"
                )
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] == "some guidance"

    def test_rename_column_sqlite_preserves_null(
        self, sqlite_backend: SQLiteStorageBackend
    ) -> None:
        """Verify migration preserves NULL values correctly."""
        from ragzoom.migrations import migrate_summary_prompt_column

        # Insert test data with NULL prompt
        with sqlite_backend.db.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO documents "
                    "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                    "VALUES ('test-doc-3', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                    "datetime('now'), NULL)"
                )
            )

        # Run migration
        migrate_summary_prompt_column(sqlite_backend.db.engine)

        # Verify NULL was preserved
        with sqlite_backend.db.engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT summarization_guidance FROM documents WHERE id = 'test-doc-3'"
                )
            )
            row = result.fetchone()
            assert row is not None
            assert row[0] is None
