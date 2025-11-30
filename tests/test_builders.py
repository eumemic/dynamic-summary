"""Test data builders for creating test fixtures easily."""

from types import SimpleNamespace

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.models import Document
from ragzoom.models import PostgresTreeNode as TreeNode


class TreeNodeBuilder:
    """Builder for creating TreeNode test data with sensible defaults."""

    def __init__(self) -> None:
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
            "level_index": 0,
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
        self, left_child_id: str | None, right_child_id: str | None = None
    ) -> "TreeNodeBuilder":
        """Set child node IDs."""
        self._data["left_child_id"] = left_child_id
        self._data["right_child_id"] = right_child_id
        return self

    def with_document(self, document_id: str) -> "TreeNodeBuilder":
        """Set the document ID."""
        self._data["document_id"] = document_id
        return self

    def with_token_count(self, token_count: int) -> "TreeNodeBuilder":
        """Set the token count."""
        self._data["token_count"] = token_count
        return self

    def with_height(self, height: int) -> "TreeNodeBuilder":
        """Set the node height."""
        self._data["height"] = height
        return self

    def build(
        self, target: str = "model"
    ) -> TreeNode | SimpleNamespace | dict[str, object]:
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
            # Add additional attributes for compatibility
            node.is_pinned = 0
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
                "level_index": self._data["level_index"],
            }
        else:
            raise ValueError(
                f"Unknown target: {target}. Use 'model', 'namespace', or 'dict'"
            )

    # Legacy methods for backward compatibility
    def build_simple_namespace(self) -> SimpleNamespace:
        """Build a SimpleNamespace for mock store compatibility. [DEPRECATED: Use build('namespace')]"""
        result = self.build("namespace")
        assert isinstance(result, SimpleNamespace)
        return result

    def build_dict(self) -> dict[str, object]:
        """Build a dict for add_batch. [DEPRECATED: Use build('dict')]"""
        result = self.build("dict")
        assert isinstance(result, dict)
        return result


class DocumentBuilder:
    """Builder for creating Document test data with sensible defaults."""

    def __init__(self) -> None:
        self._data: dict[str, object] = {
            "id": "test-doc-1",
            "file_path": "/test/document.txt",
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
            # Add additional attributes for compatibility
            doc.created_at = None
            return doc
        else:
            raise ValueError(f"Unknown target: {target}. Use 'model' or 'namespace'")

    # Legacy method for backward compatibility
    def build_simple_namespace(self) -> SimpleNamespace:
        """Build a SimpleNamespace for mock store compatibility. [DEPRECATED: Use build('namespace')]"""
        result = self.build("namespace")
        assert isinstance(result, SimpleNamespace)
        return result


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
            parent_idx = (i - 3) // 2 + 1
            if parent_idx < i:
                builder = builder.with_parent(f"node-{parent_idx}")

        # Vary text and spans
        builder = builder.with_text(f"Test text for node {i}").with_span(
            i * 10, (i + 1) * 10
        )

        result = builder.build("model")
        assert isinstance(result, TreeNode)
        nodes.append(result)

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

        result = builder.build("namespace")
        assert isinstance(result, SimpleNamespace)
        nodes.append(result)

    return nodes


def make_node_data(
    node_id: str = "test-node",
    text: str = "Test text",
    span_start: int = 0,
    span_end: int = 10,
    height: int = 0,
    token_count: int = 5,
    level_index: int = 0,
    parent_id: str | None = None,
    left_child_id: str | None = None,
    right_child_id: str | None = None,
    document_id: str | None = None,
    preceding_neighbor_id: str | None = None,
    following_neighbor_id: str | None = None,
) -> NodeDataDict:
    """Create node data dict with all required fields.

    Use this helper when creating raw node data for tests. It ensures all
    required fields have sensible defaults, avoiding KeyError when the
    repository code accesses them directly.

    Example:
        nodes = [
            make_node_data(node_id="leaf-1", text="Hello", span_start=0, span_end=5),
            make_node_data(node_id="leaf-2", text="World", span_start=5, span_end=10),
        ]
        doc_store.nodes.add_batch(nodes)
    """
    return {
        "node_id": node_id,
        "text": text,
        "span_start": span_start,
        "span_end": span_end,
        "height": height,
        "token_count": token_count,
        "level_index": level_index,
        "parent_id": parent_id,
        "left_child_id": left_child_id,
        "right_child_id": right_child_id,
        "document_id": document_id,
        "preceding_neighbor_id": preceding_neighbor_id,
        "following_neighbor_id": following_neighbor_id,
    }
