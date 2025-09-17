"""Tests for SQLAlchemy models and database schema validation."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from ragzoom.models import Base, Document
from ragzoom.models import PostgresTreeNode as TreeNode


class TestTreeNodeModel:
    """Test TreeNode SQLAlchemy model definition."""

    def test_table_name(self) -> None:
        """Test that TreeNode uses correct table name."""
        assert TreeNode.__tablename__ == "tree_nodes"

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
        assert hasattr(Document, "content_hash")
        assert hasattr(Document, "embedding_model")
        assert hasattr(Document, "summary_model")

    def test_optional_fields(self) -> None:
        """Test that Document has correct optional fields."""
        assert hasattr(Document, "file_path")
        assert hasattr(Document, "indexed_at")
        assert hasattr(Document, "chunk_count")

    def test_unique_constraints(self) -> None:
        """Test that Document has correct unique constraints."""
        file_path_col = Document.__table__.columns["file_path"]
        assert file_path_col.unique is True

    def test_default_values(self) -> None:
        """Test that Document has correct default values."""
        chunk_count_col = Document.__table__.columns["chunk_count"]
        assert chunk_count_col.default.arg == 0

    def test_primary_key(self) -> None:
        """Test that Document has correct primary key."""
        pk_columns = [col.name for col in Document.__table__.primary_key.columns]  # type: ignore[attr-defined]
        assert pk_columns == ["id"]


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
            content_hash="abc123",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        assert doc.id == "test_doc"
        assert doc.content_hash == "abc123"

    def test_datetime_fields(self) -> None:
        """Test that datetime fields work correctly."""
        # Test that datetime fields accept datetime objects
        now = datetime.utcnow()

        node = TreeNode(
            id="test_node",
            span_start=0,
            span_end=100,
            text="Test content",
            created_at=now,
            last_accessed=now,
        )

        assert node.created_at == now
        assert node.last_accessed == now

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
            content_hash="abc123",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            file_path=None,
        )

        assert doc.file_path is None


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
