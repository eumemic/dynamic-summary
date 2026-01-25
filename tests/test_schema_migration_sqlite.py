"""Tests for database schema migration functions.

Tests cover:
- detect_schema_version() identifying column names in Document table
- Migration from summary_system_prompt to summarization_guidance (future phases)
"""

import pytest
from sqlalchemy import DateTime, Integer, String, create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.pool import StaticPool

from ragzoom.migrations import SchemaVersion, detect_schema_version


def create_v1_schema_engine() -> Engine:
    """Create an engine with V1 schema (summary_system_prompt column).

    This creates a database with the old schema for testing migration paths.
    """
    engine = create_engine(
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

    V1Base.metadata.create_all(engine)
    return engine


class TestDetectSchemaVersion:
    """Tests for detect_schema_version function."""

    def test_detect_schema_version_with_old_column(self) -> None:
        """Verify detect_schema_version returns V1 when summary_system_prompt exists."""
        engine = create_v1_schema_engine()
        try:
            version = detect_schema_version(engine)
            assert version == SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT
        finally:
            engine.dispose()

    def test_detect_schema_version_with_new_column(self) -> None:
        """Verify detect_schema_version returns V2 when both columns exist.

        When both columns exist, we treat it as V2 (migration in progress).
        """
        engine = create_v1_schema_engine()
        try:
            # Add new column to simulate partially-migrated schema
            with engine.begin() as conn:
                conn.execute(
                    text("ALTER TABLE documents ADD COLUMN summarization_guidance TEXT")
                )

            version = detect_schema_version(engine)
            # When both columns exist, we treat it as V2 (new takes precedence)
            assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE
        finally:
            engine.dispose()

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

    def test_rename_column_sqlite(self) -> None:
        """Verify migrate_summary_prompt_column renames column in SQLite.

        Requirements from spec:
        - Column is renamed from summary_system_prompt to summarization_guidance
        - Existing data is preserved
        - Migration is idempotent (safe to run multiple times)
        """
        from ragzoom.migrations import migrate_summary_prompt_column

        engine = create_v1_schema_engine()
        try:
            # Verify starting state: V1 schema
            version = detect_schema_version(engine)
            assert version == SchemaVersion.V1_SUMMARY_SYSTEM_PROMPT

            # Insert test data before migration
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                        "VALUES ('test-doc-1', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                        "datetime('now'), 'test guidance')"
                    )
                )

            # Run migration
            migrate_summary_prompt_column(engine)

            # Verify column was renamed
            version = detect_schema_version(engine)
            assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

            # Verify data was preserved in new column
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT summarization_guidance FROM documents WHERE id = 'test-doc-1'"
                    )
                )
                row = result.fetchone()
                assert row is not None
                assert row[0] == "test guidance"
        finally:
            engine.dispose()

    def test_rename_column_sqlite_idempotent(self) -> None:
        """Verify migration is idempotent - safe to run multiple times."""
        from ragzoom.migrations import migrate_summary_prompt_column

        engine = create_v1_schema_engine()
        try:
            # Insert test data
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                        "VALUES ('test-doc-2', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                        "datetime('now'), 'some guidance')"
                    )
                )

            # Run migration twice
            migrate_summary_prompt_column(engine)
            migrate_summary_prompt_column(engine)

            # Should still be V2 and data intact
            version = detect_schema_version(engine)
            assert version == SchemaVersion.V2_SUMMARIZATION_GUIDANCE

            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT summarization_guidance FROM documents WHERE id = 'test-doc-2'"
                    )
                )
                row = result.fetchone()
                assert row is not None
                assert row[0] == "some guidance"
        finally:
            engine.dispose()

    def test_rename_column_sqlite_preserves_null(self) -> None:
        """Verify migration preserves NULL values correctly."""
        from ragzoom.migrations import migrate_summary_prompt_column

        engine = create_v1_schema_engine()
        try:
            # Insert test data with NULL prompt
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO documents "
                        "(id, embedding_model, summary_model, is_temporal, indexed_at, summary_system_prompt) "
                        "VALUES ('test-doc-3', 'text-embedding-3-small', 'gpt-4o-mini', 0, "
                        "datetime('now'), NULL)"
                    )
                )

            # Run migration
            migrate_summary_prompt_column(engine)

            # Verify NULL was preserved
            with engine.connect() as conn:
                result = conn.execute(
                    text(
                        "SELECT summarization_guidance FROM documents WHERE id = 'test-doc-3'"
                    )
                )
                row = result.fetchone()
                assert row is not None
                assert row[0] is None
        finally:
            engine.dispose()
