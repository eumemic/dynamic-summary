"""Storage layer for RagZoom using PostgreSQL with pgvector for embeddings."""

import hashlib
import logging
import os
from contextlib import contextmanager
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import OperationalConfig
from ragzoom.db_utils import create_temp_database, drop_temp_database, get_temp_db_name
from ragzoom.exceptions import InvalidOperationError, NodeNotFoundError
from ragzoom.models import Base, Document, TreeNode
from ragzoom.repositories.document_repository import DocumentRepository
from ragzoom.repositories.node_repository import NodeRepository
from ragzoom.services.cache_manager import CacheManager
from ragzoom.services.search_service import SearchService
from ragzoom.services.tree_navigator import TreeNavigator
from ragzoom.storage.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


def create_store_with_docker(
    config: OperationalConfig, embedding_model: str = "text-embedding-3-small"
) -> "Store":
    """Create a Store instance with automatic Docker PostgreSQL if needed.

    This factory function handles Docker container startup when using
    the default database URL without explicit configuration.

    Args:
        config: Operational configuration
        embedding_model: Name of embedding model

    Returns:
        Configured Store instance

    Raises:
        OSError: If Docker PostgreSQL startup fails
    """
    database_url = config.database_url

    # Check if we should auto-start Docker PostgreSQL
    should_auto_start = (
        database_url == "postgresql+psycopg://localhost/ragzoom"
        and not os.getenv("RAGZOOM_DATABASE_URL")  # User didn't explicitly set URL
        and not os.getenv("RAGZOOM_NO_DOCKER")  # User didn't disable Docker
    )

    if should_auto_start:
        try:
            from ragzoom.docker_postgres import DockerPostgres

            docker_pg = DockerPostgres()
            database_url = docker_pg.ensure_running()
            logger.info("✅ PostgreSQL ready in Docker container")

            # Update config with Docker database URL
            config = OperationalConfig(
                openai_api_key=config.openai_api_key,
                database_url=database_url,
                cache_size=config.cache_size,
            )
        except ImportError:
            logger.debug("Docker PostgreSQL management not available")
        except OSError:
            # User-friendly errors from DockerPostgres - re-raise as-is
            raise
        except Exception as e:
            logger.debug(f"Auto-start failed: {e}")
            raise OSError(
                f"\n❌ Failed to start PostgreSQL automatically.\n\n"
                f"Run 'ragzoom doctor' to diagnose the issue.\n"
                f"Error: {str(e)}"
            )

    return Store(config, embedding_model)


class Store:
    """Combined storage for tree structure and embeddings in PostgreSQL.

    Error Handling Contract:
    - Query methods (get_*): Return None for not found, never raise for missing items
    - Predicate methods (is_*): Return False for missing items, never raise
    - Command methods (add_*, pin_*, delete_*): Raise specific exceptions for failures
    - Calculation methods (get_node_depth): Raise NodeNotFoundError for missing nodes

    Exceptions:
    - NodeNotFoundError: When a requested node cannot be found
    - DocumentNotFoundError: When a requested document cannot be found
    - InvalidOperationError: When operation cannot be performed (validation, already exists, etc.)
    - StorageError: When storage backend encounters internal errors
    """

    # Class constants
    PIN_DEPTH_MAX = 2  # Maximum depth for pinned nodes (dormant feature)
    DEFAULT_CACHE_SIZE = 1000  # Default LRU cache size for hot nodes
    DEFAULT_POOL_SIZE = 10  # Default connection pool size
    DEFAULT_MAX_OVERFLOW = 20  # Default max overflow connections

    @classmethod
    def temporary(cls, embedding_model: str = "text-embedding-3-small"):
        """Create a temporary store for testing/benchmarking.

        Returns a context manager that yields a Store instance with a temporary
        PostgreSQL database that is automatically cleaned up.
        """

        @contextmanager
        def _temporary_store():
            # Generate unique database name
            temp_db_name = get_temp_db_name()

            try:
                # Create temporary database
                temp_db_url = create_temp_database(temp_db_name)

                # Create store configuration
                temp_config = OperationalConfig(
                    openai_api_key=os.getenv("OPENAI_API_KEY", "test-key"),
                    database_url=temp_db_url,
                )

                # Create the store
                store = cls(temp_config, embedding_model=embedding_model)

                # Create tables
                Base.metadata.create_all(store.engine)

                yield store
            finally:
                # Cleanup
                try:
                    if "store" in locals():
                        store.close()
                    drop_temp_database(temp_db_name)
                except Exception as e:
                    logger.warning(f"Failed to cleanup temporary store: {e}")

        return _temporary_store()

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

        # Transaction state tracking
        self._active_transaction = False

        # Initialize components using dependency injection
        self.db_manager = DatabaseManager(config, embedding_model)
        self.cache_manager = CacheManager[TreeNode](
            config.cache_size or self.DEFAULT_CACHE_SIZE
        )
        self.node_repo = NodeRepository(self.db_manager, self.cache_manager)
        self.doc_repo = DocumentRepository(self.db_manager, self.cache_manager)
        self.search_service = SearchService(self.db_manager)
        self.tree_navigator = TreeNavigator(self.node_repo)

        # Expose properties for backward compatibility
        self.SessionLocal = self.db_manager.SessionLocal
        self.engine = self.db_manager.engine

        # Cache properties for backward compatibility
        self.node_cache = self.cache_manager.cache
        self.cache_order = self.cache_manager.cache_order

        # Store expected embedding dimension
        self._expected_embedding_dim = self.db_manager._expected_embedding_dim

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
        """Add a node to the database with its embedding."""
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

    def add_nodes_batch(
        self, nodes_data: list[dict[str, Any]], *, session=None
    ) -> list[TreeNode]:
        """Add multiple nodes to the database in batch."""
        return self.node_repo.add_nodes_batch(nodes_data, session=session)

    def update_parent_references_batch(
        self, updates: list[tuple[str, str]], *, session=None
    ) -> None:
        """Update parent references for multiple nodes in batch."""
        self.node_repo.update_parent_references_batch(updates, session=session)

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

    def pin_node(self, node_id: str) -> None:
        """Pin a node if it's within allowed depth.

        Raises:
            NodeNotFoundError: If the node does not exist
            InvalidOperationError: If the node is too deep or already pinned
        """
        node = self.get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        node_depth = self.get_node_depth(node_id)
        if node_depth > self.PIN_DEPTH_MAX:
            raise InvalidOperationError(
                f"Node {node_id} is at depth {node_depth}, which exceeds maximum pin depth {self.PIN_DEPTH_MAX}"
            )

        # Check if already pinned
        if node.is_pinned == 1:
            raise InvalidOperationError(f"Node {node_id} is already pinned")

        self.node_repo.pin_node(node_id)

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes with no children)."""
        return self.node_repo.get_leaf_nodes()

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        """Get all nodes for a specific document."""
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

    # jscpd:ignore - Delegation method with same signature as repository
    def add_document(
        self,
        document_id: str,
        file_path: str | None,
        content_hash: str,
        chunk_count: int,
        embedding_model: str,
        summary_model: str,
        *,
        session=None,
    ) -> Document:
        """Add a document record."""
        return self.doc_repo.add_document(
            document_id,
            file_path,
            content_hash,
            chunk_count,
            embedding_model,
            summary_model,
            session=session,
        )

    def delete_document_nodes(self, document_id: str, *, session=None) -> int:
        """Delete all nodes associated with a document."""
        return self.doc_repo.delete_document_nodes(document_id, session=session)

    def clear_document(self, document_id: str, *, session=None) -> int:
        """Clear all data for a document, including orphaned nodes and document record."""
        return self.doc_repo.clear_document(document_id, session=session)

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
        """Search for similar nodes using pgvector cosine distance."""
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
    @contextmanager
    def transaction(self):
        """Context manager for transactional operations.

        Usage:
            with store.transaction() as session:
                store.add_document(..., session=session)
                store.add_nodes_batch(..., session=session)
                # All operations commit together or all rollback

        Yields:
            SQLAlchemy session for the transaction

        Raises:
            RuntimeError: If nested transaction is attempted
            Any exception from the transactional operations (after rollback)
        """
        if self._active_transaction:
            raise RuntimeError(
                "Nested transactions are not supported. "
                "Please use the same session for all operations within a transaction."
            )

        self._active_transaction = True
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._active_transaction = False
            session.close()

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        self.db_manager.close()

    # Static utility methods
    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    # Deprecated methods - included for backward compatibility but delegate to new components
    def _run_migrations(self) -> None:
        """Run database migrations (deprecated - handled by DatabaseManager)."""
        # This is now handled by DatabaseManager during initialization
        pass
