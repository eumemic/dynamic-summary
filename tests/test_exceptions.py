"""Tests for RagZoom domain-specific exceptions."""

import pytest

from ragzoom.exceptions import (
    DocumentNotFoundError,
    InvalidOperationError,
    NodeNotFoundError,
    StorageError,
)


class TestNodeNotFoundError:
    """Test NodeNotFoundError exception."""

    def test_message_includes_node_id(self):
        """Test that error message includes the node ID."""
        node_id = "test_node_123"
        error = NodeNotFoundError(node_id)

        assert str(error) == f"Node {node_id} not found"
        assert error.node_id == node_id

    def test_inheritance(self):
        """Test that NodeNotFoundError inherits from Exception."""
        error = NodeNotFoundError("test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught properly."""
        with pytest.raises(NodeNotFoundError) as exc_info:
            raise NodeNotFoundError("missing_node")

        assert exc_info.value.node_id == "missing_node"


class TestDocumentNotFoundError:
    """Test DocumentNotFoundError exception."""

    def test_message_includes_document_id(self):
        """Test that error message includes the document ID."""
        doc_id = "test_doc_456"
        error = DocumentNotFoundError(doc_id)

        assert str(error) == f"Document {doc_id} not found"
        assert error.document_id == doc_id

    def test_inheritance(self):
        """Test that DocumentNotFoundError inherits from Exception."""
        error = DocumentNotFoundError("test")
        assert isinstance(error, Exception)

    def test_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught properly."""
        with pytest.raises(DocumentNotFoundError) as exc_info:
            raise DocumentNotFoundError("missing_doc")

        assert exc_info.value.document_id == "missing_doc"


class TestInvalidOperationError:
    """Test InvalidOperationError exception."""

    def test_inheritance(self):
        """Test that InvalidOperationError inherits from Exception."""
        error = InvalidOperationError("test message")
        assert isinstance(error, Exception)

    def test_custom_message(self):
        """Test that custom error messages work correctly."""
        message = "Cannot perform operation: invalid state"
        error = InvalidOperationError(message)
        assert str(error) == message

    def test_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught properly."""
        with pytest.raises(InvalidOperationError):
            raise InvalidOperationError("Invalid operation attempted")


class TestStorageError:
    """Test StorageError exception."""

    def test_inheritance(self):
        """Test that StorageError inherits from Exception."""
        error = StorageError("test message")
        assert isinstance(error, Exception)

    def test_custom_message(self):
        """Test that custom error messages work correctly."""
        message = "Database connection failed"
        error = StorageError(message)
        assert str(error) == message

    def test_can_be_raised_and_caught(self):
        """Test that exception can be raised and caught properly."""
        with pytest.raises(StorageError):
            raise StorageError("Storage operation failed")


class TestExceptionInteraction:
    """Test exception interactions and edge cases."""

    def test_different_exceptions_are_distinct(self):
        """Test that different exception types can be caught separately."""
        # Test that NodeNotFoundError doesn't catch DocumentNotFoundError
        with pytest.raises(DocumentNotFoundError):
            try:
                raise DocumentNotFoundError("test_doc")
            except NodeNotFoundError:
                pytest.fail("Should not catch NodeNotFoundError")

    def test_all_exceptions_caught_by_base_exception(self):
        """Test that all custom exceptions are caught by base Exception."""
        exceptions = [
            NodeNotFoundError("node"),
            DocumentNotFoundError("doc"),
            InvalidOperationError("invalid"),
            StorageError("storage"),
        ]

        for exc in exceptions:
            with pytest.raises(Exception):
                raise exc
