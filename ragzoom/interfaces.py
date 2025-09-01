"""Store interface protocol for type safety and testing."""

# jscpd:ignore-start - Protocol interfaces legitimately duplicate method signatures from implementations
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict

from ragzoom.models import Document, TreeNode


class NodeData(TypedDict, total=False):
    """Data structure for batch node creation."""

    # Required fields
    node_id: str
    text: str
    embedding: list[float] | NDArray[np.float64]
    span_start: int
    span_end: int

    # Optional fields
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    document_id: str | None
    token_count: int
    height: int
    is_left_child: bool | None


class SearchMetadata(TypedDict):
    """Metadata returned by search operations."""

    # Fields actually returned by search_similar
    span_start: int
    span_end: int
    parent_id: str  # Never None in actual return
    document_id: str  # Never None in actual return
    is_leaf: int


@runtime_checkable
class StoreInterface(Protocol):
    """Protocol defining the Store interface for type safety and testing.

    This protocol ensures that both the real Store and test mocks provide
    the same interface, enabling safe substitution in tests.
    """

    # Node operations
    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: list[float] | NDArray[np.float64],
        span_start: int,
        span_end: int,
        parent_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
        document_id: str | None = None,
        token_count: int = 0,
        height: int = 0,
        is_left_child: bool | None = None,
    ) -> TreeNode:
        """Add a node to the store."""
        ...

    def add_nodes_batch(self, nodes_data: list[NodeData]) -> list[TreeNode]:
        """Add multiple nodes in batch."""
        ...

    def update_parent_references_batch(self, updates: list[tuple[str, str]]) -> None:
        """Update parent references for multiple nodes."""
        ...

    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID."""
        ...

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by their IDs."""
        ...

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node."""
        ...

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get all pinned nodes up to optional max depth."""
        ...

    def pin_node(self, node_id: str) -> None:
        """Pin a node."""
        ...

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes."""
        ...

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        """Get all nodes for a specific document."""
        ...

    def get_all_nodes_for_document_paginated(
        self, document_id: str | None, *, page_size: int = 1000
    ) -> list[list[TreeNode]]:
        """Get all nodes for a document in paginated batches for memory efficiency."""
        ...

    # Document operations
    def get_document_by_path(self, file_path: str) -> Document | None:
        """Get a document by file path."""
        ...

    def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        ...

    def get_document_embedding_model(self, document_id: str) -> str | None:
        """Get the embedding model used for a specific document."""
        ...

    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
    ) -> Document:
        """Add a document record."""
        ...

    def delete_document_nodes(self, document_id: str, *, session: None = None) -> int:
        """Delete all nodes associated with a document."""
        ...

    def clear_document(self, document_id: str) -> int:
        """Clear all data for a document."""
        ...

    def get_document_token_stats(self, document_id: str) -> dict[str, float | int]:
        """Get token statistics for a document."""
        ...

    # Search operations
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, str | int | float | bool | None] | None = None,
    ) -> list[tuple[str, float, SearchMetadata]]:
        """Search for similar nodes."""
        ...

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, SearchMetadata]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Apply MMR to get diverse results."""
        ...

    # Tree navigation operations
    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        """Get left and right children of a node."""
        ...

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get all ancestors of given nodes."""
        ...

    def get_root_node(self) -> TreeNode | None:
        """Get the root node."""
        ...

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        """Get the root node for a specific document."""
        ...

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node."""
        ...

    def is_leaf_node(self, node_id: str) -> bool:
        """Check if a node is a leaf."""
        ...

    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is a root."""
        ...

    # Utility methods
    def close(self) -> None:
        """Close database connections and cleanup resources."""
        ...

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        ...


# jscpd:ignore-end
