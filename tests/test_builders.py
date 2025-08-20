"""Test data builders for creating test fixtures easily."""

from types import SimpleNamespace
from typing import Any

from ragzoom.models import Document, TreeNode


class TreeNodeBuilder:
    """Builder for creating TreeNode test data with sensible defaults."""

    def __init__(self):
        self._data = {
            "id": "test-node-1",
            "text": "Test node text",
            "embedding": [0.1] * 1536,
            "span_start": 0,
            "span_end": 10,
            "parent_id": None,
            "left_child_id": None,
            "right_child_id": None,
            "document_id": None,
            "token_count": 5,
            "height": 0,
        }

    def with_id(self, node_id: str) -> "TreeNodeBuilder":
        """Set the node ID."""
        self._data["id"] = node_id
        return self

    def with_text(self, text: str) -> "TreeNodeBuilder":
        """Set the node text."""
        self._data["text"] = text
        return self

    def with_embedding(self, embedding: list[float]) -> "TreeNodeBuilder":
        """Set the embedding vector."""
        self._data["embedding"] = embedding
        return self

    def with_span(self, start: int, end: int) -> "TreeNodeBuilder":
        """Set the text span."""
        self._data["span_start"] = start
        self._data["span_end"] = end
        return self

    def with_parent(self, parent_id: str) -> "TreeNodeBuilder":
        """Set the parent node ID."""
        self._data["parent_id"] = parent_id
        return self

    def with_children(
        self, left_child_id: str | None = None, right_child_id: str | None = None
    ) -> "TreeNodeBuilder":
        """Set the child node IDs."""
        self._data["left_child_id"] = left_child_id
        self._data["right_child_id"] = right_child_id
        return self

    def with_document(self, document_id: str) -> "TreeNodeBuilder":
        """Set the document ID."""
        self._data["document_id"] = document_id
        return self

    def with_token_count(self, count: int) -> "TreeNodeBuilder":
        """Set the token count."""
        self._data["token_count"] = count
        return self

    def with_height(self, height: int) -> "TreeNodeBuilder":
        """Set the node height."""
        self._data["height"] = height
        return self

    def build(
        self, target: str = "model"
    ) -> TreeNode | SimpleNamespace | dict[str, Any]:
        """Build the node in the specified format.

        Args:
            target: Format to build - "model" for TreeNode, "namespace" for SimpleNamespace, "dict" for dictionary

        Returns:
            TreeNode, SimpleNamespace, or dict based on target parameter
        """
        if target == "model":
            return TreeNode(**self._data)
        elif target == "namespace":
            node = SimpleNamespace(**self._data)
            # Add additional attributes that SimpleMockStore expects
            node.is_pinned = 0
            node.last_accessed = None
            node.access_count = 0
            node.created_at = None
            node.preceding_neighbor_id = None
            return node
        elif target == "dict":
            return {
                "node_id": self._data["id"],
                "text": self._data["text"],
                "embedding": self._data["embedding"],
                "span_start": self._data["span_start"],
                "span_end": self._data["span_end"],
                "parent_id": self._data["parent_id"],
                "left_child_id": self._data["left_child_id"],
                "right_child_id": self._data["right_child_id"],
                "document_id": self._data["document_id"],
                "token_count": self._data["token_count"],
                "height": self._data["height"],
            }
        else:
            raise ValueError(
                f"Unknown target: {target}. Use 'model', 'namespace', or 'dict'"
            )

    # Legacy methods for backward compatibility
    def build_simple_namespace(self) -> SimpleNamespace:
        """Build a SimpleNamespace for mock store compatibility. [DEPRECATED: Use build('namespace')]"""
        return self.build("namespace")

    def build_dict(self) -> dict[str, Any]:
        """Build a dictionary for batch operations. [DEPRECATED: Use build('dict')]"""
        return self.build("dict")


class DocumentBuilder:
    """Builder for creating Document test data with sensible defaults."""

    def __init__(self):
        self._data = {
            "id": "test-doc-1",
            "file_path": "/test/document.txt",
            "content_hash": "abc123",
            "chunk_count": 5,
            "embedding_model": "text-embedding-3-small",
            "summary_model": "gpt-4o-mini",
        }

    def with_id(self, document_id: str) -> "DocumentBuilder":
        """Set the document ID."""
        self._data["id"] = document_id
        return self

    def with_file_path(self, file_path: str | None) -> "DocumentBuilder":
        """Set the file path."""
        self._data["file_path"] = file_path
        return self

    def with_content_hash(self, content_hash: str) -> "DocumentBuilder":
        """Set the content hash."""
        self._data["content_hash"] = content_hash
        return self

    def with_chunk_count(self, count: int) -> "DocumentBuilder":
        """Set the chunk count."""
        self._data["chunk_count"] = count
        return self

    def with_embedding_model(self, model: str) -> "DocumentBuilder":
        """Set the embedding model."""
        self._data["embedding_model"] = model
        return self

    def with_summary_model(self, model: str) -> "DocumentBuilder":
        """Set the summary model."""
        self._data["summary_model"] = model
        return self

    def build(self, target: str = "model") -> Document | SimpleNamespace:
        """Build the document in the specified format.

        Args:
            target: Format to build - "model" for Document, "namespace" for SimpleNamespace

        Returns:
            Document or SimpleNamespace based on target parameter
        """
        if target == "model":
            return Document(**self._data)
        elif target == "namespace":
            doc = SimpleNamespace(**self._data)
            # Add additional attributes that SimpleMockStore expects
            doc.created_at = None
            return doc
        else:
            raise ValueError(f"Unknown target: {target}. Use 'model' or 'namespace'")

    # Legacy method for backward compatibility
    def build_simple_namespace(self) -> SimpleNamespace:
        """Build a SimpleNamespace for mock store compatibility. [DEPRECATED: Use build('namespace')]"""
        return self.build("namespace")


def create_test_tree_nodes(
    count: int = 3, document_id: str = "test-doc"
) -> list[TreeNode]:
    """Create a simple test tree with the specified number of nodes.

    Creates a binary tree structure:
    - Node 0: Root (parent of nodes 1 and 2)
    - Node 1: Left child of root
    - Node 2: Right child of root
    - Additional nodes as children of previous nodes
    """
    nodes = []

    for i in range(count):
        builder = TreeNodeBuilder().with_id(f"node-{i}").with_document(document_id)

        if i == 0:
            # Root node
            if count > 1:
                builder = builder.with_children(
                    "node-1", "node-2" if count > 2 else None
                )
        elif i == 1:
            # Left child of root
            builder = builder.with_parent("node-0")
        elif i == 2:
            # Right child of root
            builder = builder.with_parent("node-0")
        else:
            # Additional nodes as children of previous nodes
            parent_idx = (i - 3) // 2 + 1  # Distribute among non-root nodes
            if parent_idx < i:
                builder = builder.with_parent(f"node-{parent_idx}")

        # Vary text and spans
        builder = builder.with_text(f"Test text for node {i}").with_span(
            i * 10, (i + 1) * 10
        )

        nodes.append(builder.build())

    return nodes


def create_simple_namespace_tree(
    count: int = 3, document_id: str = "test-doc"
) -> list[SimpleNamespace]:
    """Create a simple test tree using SimpleNamespace objects for mock store."""
    nodes = []

    for i in range(count):
        builder = TreeNodeBuilder().with_id(f"node-{i}").with_document(document_id)

        if i == 0:
            # Root node
            if count > 1:
                builder = builder.with_children(
                    "node-1", "node-2" if count > 2 else None
                )
        elif i == 1:
            # Left child of root
            builder = builder.with_parent("node-0")
        elif i == 2:
            # Right child of root
            builder = builder.with_parent("node-0")
        else:
            # Additional nodes as children of previous nodes
            parent_idx = (i - 3) // 2 + 1
            if parent_idx < i:
                builder = builder.with_parent(f"node-{parent_idx}")

        # Vary text and spans
        builder = builder.with_text(f"Test text for node {i}").with_span(
            i * 10, (i + 1) * 10
        )

        nodes.append(builder.build("namespace"))

    return nodes
