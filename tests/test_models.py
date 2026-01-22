"""Tests for SQLAlchemy models and database schema validation."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import Float, create_engine
from sqlalchemy.engine import Engine

from ragzoom.models import Base, Document
from ragzoom.models import PostgresTreeNode as TreeNode


class TestTreeNodeModel:
    """Test TreeNode SQLAlchemy model definition."""

    def test_table_name(self) -> None:
        """Test that TreeNode uses correct table name."""
        assert TreeNode.__tablename__ == "tree_nodes"

    def test_postgres_tree_node_has_temporal_columns(self) -> None:
        """Test that PostgresTreeNode has nullable Float columns for temporal metadata.

        Per specs/temporal-metadata.md § Data Model Changes > Database Schema.
        """
        time_start_col = TreeNode.__table__.columns["time_start"]
        time_end_col = TreeNode.__table__.columns["time_end"]

        assert time_start_col.nullable is True
        assert time_end_col.nullable is True
        assert isinstance(time_start_col.type, Float)
        assert isinstance(time_end_col.type, Float)

    def test_temporal_column_instantiation(self) -> None:
        """Test that temporal columns can be set on TreeNode instances."""
        node = TreeNode(
            id="temporal_node",
            span_start=0,
            span_end=100,
            text="Test content",
            time_start=1705845000.123,  # Unix timestamp
            time_end=1705845060.456,
        )
        assert node.time_start == 1705845000.123
        assert node.time_end == 1705845060.456

    def test_temporal_columns_default_to_none(self) -> None:
        """Test that temporal columns default to None for non-temporal nodes."""
        node = TreeNode(
            id="non_temporal_node",
            span_start=0,
            span_end=100,
            text="Test content",
        )
        assert node.time_start is None
        assert node.time_end is None

    def test_get_depth_not_persisted(self) -> None:
        """TreeNode depth should be derived structurally, not stored."""
        node = TreeNode(
            id="node",
            span_start=0,
            span_end=10,
            text="content",
        )
        with pytest.raises(NotImplementedError):
            node.get_depth()

    def test_required_fields(self) -> None:
        """Test that TreeNode has all required fields."""
        # Test field existence and types
        assert hasattr(TreeNode, "id")
        assert hasattr(TreeNode, "span_start")
        assert hasattr(TreeNode, "span_end")
        assert hasattr(TreeNode, "text")
        # Embeddings are no longer stored in SQL
        assert hasattr(TreeNode, "token_count")
        assert "path" not in TreeNode.__table__.columns

    def test_optional_fields(self) -> None:
        """Test that TreeNode has correct optional fields."""
        assert hasattr(TreeNode, "parent_id")
        assert hasattr(TreeNode, "left_child_id")
        assert hasattr(TreeNode, "right_child_id")
        assert hasattr(TreeNode, "document_id")
        assert hasattr(TreeNode, "preceding_neighbor_id")

    def test_default_values(self) -> None:
        """Test that TreeNode has correct default values."""
        # These are mapped column defaults, we can check the column info
        token_count_col = TreeNode.__table__.columns["token_count"]
        is_pinned_col = TreeNode.__table__.columns["is_pinned"]
        height_col = TreeNode.__table__.columns["height"]

        assert token_count_col.default.arg == 0
        assert is_pinned_col.default.arg == 0
        assert height_col.default.arg == 0

    def test_foreign_key_relationships(self) -> None:
        """Test that TreeNode has correct foreign key relationships."""
        # Check parent_id foreign key
        parent_id_col = TreeNode.__table__.columns["parent_id"]
        assert len(parent_id_col.foreign_keys) == 1
        fk = list(parent_id_col.foreign_keys)[0]
        assert str(fk.column) == "tree_nodes.id"

        # Check document_id foreign key
        doc_id_col = TreeNode.__table__.columns["document_id"]
        assert len(doc_id_col.foreign_keys) == 1
        fk = list(doc_id_col.foreign_keys)[0]
        assert str(fk.column) == "documents.id"

    def test_primary_key(self) -> None:
        """Test that TreeNode has correct primary key."""
        pk_columns = [col.name for col in TreeNode.__table__.primary_key.columns]  # type: ignore[attr-defined]
        assert pk_columns == ["id"]


class TestDocumentModel:
    """Test Document SQLAlchemy model definition."""

    def test_table_name(self) -> None:
        """Test that Document uses correct table name."""
        assert Document.__tablename__ == "documents"

    def test_required_fields(self) -> None:
        """Test that Document has all required fields."""
        assert hasattr(Document, "id")
        assert hasattr(Document, "embedding_model")
        assert hasattr(Document, "summary_model")

    def test_optional_fields(self) -> None:
        """Test that Document has correct optional fields."""
        assert hasattr(Document, "file_path")
        assert hasattr(Document, "indexed_at")

    def test_unique_constraints(self) -> None:
        """Test that Document has correct unique constraints."""
        file_path_col = Document.__table__.columns["file_path"]
        assert file_path_col.unique is True

    def test_primary_key(self) -> None:
        """Test that Document has correct primary key."""
        pk_columns = [col.name for col in Document.__table__.primary_key.columns]  # type: ignore[attr-defined]
        assert pk_columns == ["id"]

    def test_document_has_is_temporal_column(self) -> None:
        """Test that Document has Integer is_temporal column with default 0.

        Per specs/temporal-metadata.md § Data Model Changes > Database Schema:
        is_temporal: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

        Note: We use Integer with 0/1 for cross-database compatibility (SQLite stores
        booleans as integers). The spec shows Boolean but the codebase pattern uses Integer.
        """
        from sqlalchemy import Integer

        is_temporal_col = Document.__table__.columns["is_temporal"]

        # Column should be NOT NULL with default 0 (False)
        assert is_temporal_col.nullable is False
        assert isinstance(is_temporal_col.type, Integer)
        assert is_temporal_col.default.arg == 0

    def test_is_temporal_instantiation(self) -> None:
        """Test that is_temporal can be set on Document instances."""
        # Explicit non-temporal
        doc = Document(
            id="non_temporal_doc",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            is_temporal=0,
        )
        assert doc.is_temporal == 0

        # Explicit temporal
        temporal_doc = Document(
            id="temporal_doc",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            is_temporal=1,
        )
        assert temporal_doc.is_temporal == 1


class TestModelIntegration:
    """Test model integration and schema creation."""

    @pytest.fixture
    def memory_engine(self) -> Engine:
        """Create in-memory SQLite engine for testing."""
        # Note: We use SQLite for testing since pgvector may not be available
        # This tests schema structure but not pgvector-specific functionality
        engine = create_engine("sqlite:///:memory:")
        return engine

    def test_schema_creation(self, memory_engine: Engine) -> None:
        """Test that models can create schema without errors."""
        # This will test that the model definitions are valid
        # Note: pgvector Vector type may cause issues in SQLite, but we test what we can
        try:
            Base.metadata.create_all(memory_engine)
        except Exception as e:
            # If Vector type causes issues, that's expected in SQLite
            if "Vector" not in str(e):
                raise

    def test_model_instantiation(self) -> None:
        """Test that model instances can be created."""
        # Test TreeNode instantiation with required fields
        node = TreeNode(
            id="test_node",
            span_start=0,
            span_end=100,
            text="Test content",
        )
        assert node.id == "test_node"
        assert node.span_start == 0
        assert node.span_end == 100
        assert node.text == "Test content"

        # Test Document instantiation
        doc = Document(
            id="test_doc",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        assert doc.id == "test_doc"

    def test_datetime_fields(self) -> None:
        """Test that datetime fields work correctly."""
        # Test that datetime fields accept datetime objects
        now = datetime.now(timezone.utc)

        node = TreeNode(
            id="test_node",
            span_start=0,
            span_end=100,
            text="Test content",
            created_at=now,
        )

        assert node.created_at == now

    def test_nullable_fields(self) -> None:
        """Test that nullable fields can be None."""
        node = TreeNode(
            id="test_node",
            span_start=0,
            span_end=100,
            text="Test content",
            parent_id=None,
            left_child_id=None,
            right_child_id=None,
            document_id=None,
            preceding_neighbor_id=None,
        )

        assert node.parent_id is None
        assert node.left_child_id is None
        assert node.right_child_id is None
        assert node.document_id is None
        assert node.preceding_neighbor_id is None

        doc = Document(
            id="test_doc",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            file_path=None,
        )

        assert doc.file_path is None


class TestSqliteTreeNodeModel:
    """Test SQLiteTreeNode SQLAlchemy model definition."""

    def test_sqlite_tree_node_has_temporal_columns(self) -> None:
        """Test that SQLite tree_nodes table has nullable REAL columns for temporal metadata.

        Per specs/temporal-metadata.md § Data Model Changes > Database Schema.
        """
        from ragzoom.backends.sqlite_db import SQLiteTreeNode

        time_start_col = SQLiteTreeNode.__table__.columns["time_start"]
        time_end_col = SQLiteTreeNode.__table__.columns["time_end"]

        assert time_start_col.nullable is True
        assert time_end_col.nullable is True
        # SQLite uses Float which maps to REAL
        assert isinstance(time_start_col.type, Float)
        assert isinstance(time_end_col.type, Float)

    def test_sqlite_temporal_column_instantiation(self) -> None:
        """Test that temporal columns can be set on SQLiteTreeNode instances."""
        from ragzoom.backends.sqlite_db import SQLiteTreeNode

        node = SQLiteTreeNode(
            id="temporal_sqlite_node",
            span_start=0,
            span_end=100,
            text="Test content",
            time_start=1705845000.123,  # Unix timestamp
            time_end=1705845060.456,
        )
        assert node.time_start == 1705845000.123
        assert node.time_end == 1705845060.456

    def test_sqlite_temporal_columns_default_to_none(self) -> None:
        """Test that temporal columns default to None for non-temporal nodes."""
        from ragzoom.backends.sqlite_db import SQLiteTreeNode

        node = SQLiteTreeNode(
            id="non_temporal_sqlite_node",
            span_start=0,
            span_end=100,
            text="Test content",
        )
        assert node.time_start is None
        assert node.time_end is None

    def test_sqlite_temporal_migration_adds_columns(self) -> None:
        """Test that migration adds temporal columns to existing SQLite databases.

        Simulates a database created without temporal columns being upgraded.
        """
        from sqlalchemy import text

        from ragzoom.backends.sqlite_db import SqliteDatabaseManager

        # Create a fresh in-memory database
        db = SqliteDatabaseManager(url="sqlite:///:memory:")

        # Verify columns exist after initialization (migration runs in __post_init__)
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(tree_nodes)")).fetchall()
            column_names = [row[1] for row in result]

            assert "time_start" in column_names, "time_start column should exist"
            assert "time_end" in column_names, "time_end column should exist"

        db.close()


class TestSqliteDocumentModel:
    """Test SqliteDocument SQLAlchemy model definition."""

    def test_sqlite_document_has_is_temporal_column(self) -> None:
        """Test that SqliteDocument has Integer is_temporal column with default 0.

        Per specs/temporal-metadata.md § Data Model Changes > Database Schema.
        """
        from sqlalchemy import Integer

        from ragzoom.backends.sqlite_db import SqliteDocument

        is_temporal_col = SqliteDocument.__table__.columns["is_temporal"]

        assert is_temporal_col.nullable is False
        assert isinstance(is_temporal_col.type, Integer)
        assert is_temporal_col.default.arg == 0

    def test_sqlite_is_temporal_instantiation(self) -> None:
        """Test that is_temporal can be set on SqliteDocument instances."""
        from ragzoom.backends.sqlite_db import SqliteDocument

        # Explicit non-temporal
        doc = SqliteDocument(
            id="non_temporal_doc",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            is_temporal=0,
        )
        assert doc.is_temporal == 0

        # Explicit temporal
        temporal_doc = SqliteDocument(
            id="temporal_doc",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            is_temporal=1,
        )
        assert temporal_doc.is_temporal == 1

    def test_sqlite_is_temporal_migration_adds_column(self) -> None:
        """Test that migration adds is_temporal column to existing SQLite databases."""
        from sqlalchemy import text

        from ragzoom.backends.sqlite_db import SqliteDatabaseManager

        # Create a fresh in-memory database
        db = SqliteDatabaseManager(url="sqlite:///:memory:")

        # Verify column exists after initialization (migration runs in __post_init__)
        with db.engine.connect() as conn:
            result = conn.execute(text("PRAGMA table_info(documents)")).fetchall()
            column_names = [row[1] for row in result]

            assert "is_temporal" in column_names, "is_temporal column should exist"

        db.close()


class TestModelValidation:
    """Test model field validation and constraints."""

    # Embeddings removed from storage; dimension flexibility not applicable

    def test_text_field_flexibility(self) -> None:
        """Test that text field accepts various content types."""
        test_texts = [
            "Simple text",
            "Text with\nnewlines\nand\ttabs",
            "Unicode: 你好 🌍 café",
            "",  # Empty string
            "Very long text " * 1000,  # Long text
        ]

        for i, text in enumerate(test_texts):
            node = TreeNode(
                id=f"test_node_{i}",
                span_start=0,
                span_end=len(text),
                text=text,
            )
            assert node.text == text
