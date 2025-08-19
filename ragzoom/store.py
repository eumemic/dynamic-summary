"""Storage layer for RagZoom - SQLite for tree structure, Chroma for vectors."""

import hashlib
import logging
from contextlib import contextmanager
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import OperationalConfig
from ragzoom.models import Document, TreeNode
from ragzoom.repositories.document_repository import DocumentRepository
from ragzoom.repositories.node_repository import NodeRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.services.search_service import SearchService
from ragzoom.services.tree_navigator import TreeNavigator
from ragzoom.storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


class Store:
    """Combined storage for tree structure (SQLite) and embeddings (Chroma)."""

    # Class constant for pin depth limit (dormant feature)
    PIN_DEPTH_MAX = 2

    def __init__(
        self, config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
    ):
        """Initialize storage backends.

        Args:
            config: Operational configuration with storage paths
            embedding_model: Name of embedding model (for dimension validation)
        """
        self.config = config
        self.embedding_model = embedding_model

        # Initialize components using dependency injection
        self.db_manager = DatabaseManager(config, embedding_model)
        self.cache_manager = CacheManager[TreeNode](config.cache_size)
        self.node_repo = NodeRepository(self.db_manager, self.cache_manager)
        self.doc_repo = DocumentRepository(self.db_manager, self.cache_manager)
        self.search_service = SearchService(self.db_manager)
        self.tree_navigator = TreeNavigator(self.node_repo)

        # Expose properties for backward compatibility
        self.SessionLocal = self.db_manager.SessionLocal
        self.collection = self.db_manager.collection
        self.engine = self.db_manager.engine
        self.chroma_client = self.db_manager.chroma_client

        # Cache properties for backward compatibility
        self.node_cache = self.cache_manager.cache
        self.cache_order = self.cache_manager.cache_order

    # Node operations - delegate to NodeRepository
    # jscpd:ignore-start
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
    ) -> TreeNode:
        """Add a node to both SQLite and Chroma."""
        return self.node_repo.add_node(
            node_id,
            text,
            embedding,
            span_start,
            span_end,
            parent_id,
            left_child_id,
            right_child_id,
            document_id,
            token_count,
            height,
        )
    # jscpd:ignore-end

    def add_nodes_batch(self, nodes_data: list[dict[str, Any]]) -> list[TreeNode]:
        """Add multiple nodes to both SQLite and Chroma in batch."""
        return self.node_repo.add_nodes_batch(nodes_data)

    def update_parent_references_batch(self, updates: list[tuple[str, str]]) -> None:
        """Update parent references for multiple nodes in batch."""
        self.node_repo.update_parent_references_batch(updates)

    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID."""
        return self.node_repo.get_node(node_id)

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by their IDs."""
        return self.node_repo.get_nodes(node_ids)

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node."""
        self.node_repo.update_node_access(node_id)

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get all pinned nodes up to optional max depth."""
        return self.node_repo.get_pinned_nodes(depth_max)

    def pin_node(self, node_id: str) -> bool:
        """Pin a node (mark as important)."""
        node = self.get_node(node_id)
        if not node:
            return False

        node_depth = self.get_node_depth(node_id)
        if node_depth > self.PIN_DEPTH_MAX:
            return False

        # Check if already pinned
        if node.is_pinned == 1:
            logger.info(f"Node {node_id} is already pinned")
            return False

        return self.node_repo.pin_node(node_id)

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes with no children)."""
        return self.node_repo.get_leaf_nodes()

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        """Get all nodes for a document."""
        return self.node_repo.get_all_nodes_for_document(document_id)

    # Document operations - delegate to DocumentRepository
    def get_document_by_path(self, file_path: str) -> Document | None:
        """Get a document by file path."""
        return self.doc_repo.get_document_by_path(file_path)

    def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        return self.doc_repo.get_document_by_id(document_id)

    def get_document_embedding_model(self, document_id: str) -> str | None:
        """Get the embedding model used for a specific document."""
        return self.doc_repo.get_document_embedding_model(document_id)

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
        return self.doc_repo.add_document(
            document_id,
            file_path,
            content_hash,
            chunk_count,
            embedding_model,
            summary_model,
        )

    def delete_document_nodes(self, document_id: str) -> int:
        """Delete all nodes associated with a document."""
        return self.doc_repo.delete_document_nodes(document_id)

    def clear_document(self, document_id: str) -> int:
        """Clear all data for a document, including orphaned nodes and document record."""
        return self.doc_repo.clear_document(document_id)

    def get_document_token_stats(self, document_id: str) -> dict[str, float | int]:
        """Get token statistics for a document using efficient SQL aggregation."""
        return self.doc_repo.get_document_token_stats(document_id)

    # Search operations - delegate to SearchService
    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar nodes using Chroma."""
        return self.search_service.search_similar(query_embedding, n_results, where)

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, dict[str, Any]]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Apply MMR (Maximal Marginal Relevance) to get diverse results."""
        return self.search_service.compute_mmr_diverse_results(
            query_embedding, candidates, lambda_param, k
        )

    # Tree navigation operations - delegate to TreeNavigator
    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        """Get left and right children of a node."""
        return self.tree_navigator.get_children(node_id)

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get all ancestors of given nodes using batch loading for efficiency."""
        return self.tree_navigator.get_ancestors(node_ids)

    def get_root_node(self) -> TreeNode | None:
        """Get the root node (node with no parent)."""
        return self.tree_navigator.get_root_node()

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        """Get the root node for a specific document."""
        return self.tree_navigator.get_root_node_for_document(document_id)

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node (distance from root)."""
        return self.tree_navigator.get_node_depth(node_id)

    def is_leaf_node(self, node_id: str) -> bool:
        """Check if a node is a leaf (has no children)."""
        return self.tree_navigator.is_leaf_node(node_id)

    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is a root (has no parent)."""
        return self.tree_navigator.is_root_node(node_id)

    # Cache operations - delegate to CacheManager
    def _get_from_cache(self, node_id: str) -> TreeNode | None:
        """Get item from cache (for backward compatibility)."""
        return self.cache_manager.get(node_id)

    def _add_to_cache(self, node: TreeNode) -> None:
        """Add item to cache (for backward compatibility)."""
        self.cache_manager.put(node.id, node)

    # Database validation - delegate to DatabaseManager
    def _validate_embedding_dimension(
        self, embedding: list[float] | NDArray[np.float64]
    ) -> None:
        """Validate that embedding has correct dimension."""
        self.db_manager.validate_embedding_dimension(embedding)

    def _get_expected_embedding_dimension(self) -> int | None:
        """Get expected embedding dimension from existing embeddings."""
        return self.db_manager._get_expected_embedding_dimension()

    # Lifecycle methods
    def close(self) -> None:
        """Close database connections."""
        self.db_manager.close()

    # Static utility methods
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    @staticmethod
    @contextmanager
    def temporary():
        """Create a temporary in-memory store for testing."""
        config = OperationalConfig(
            sqlite_database_url="sqlite:///:memory:",
            chroma_persist_directory=":memory:",
            cache_size=100,
        )
        store = Store(config)
        try:
            yield store
        finally:
            store.close()

    # Deprecated methods - included for backward compatibility but delegate to new components
    def _run_migrations(self) -> None:
        """Run database migrations (deprecated - handled by DatabaseManager)."""
        # This is now handled by DatabaseManager during initialization
        pass
