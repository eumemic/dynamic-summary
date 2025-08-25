"""Storage layer for RagZoom using PostgreSQL with pgvector for embeddings."""

import hashlib
import logging
import os
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from ragzoom.config import OperationalConfig, SecretStr
from ragzoom.db_utils import create_temp_database, drop_temp_database, get_temp_db_name
from ragzoom.document_store import DocumentStore
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
) -> "StoreManager":
    """Create a Store instance with automatic Docker PostgreSQL if needed.

    This factory function handles Docker container startup when using
    the default database URL without explicit configuration.

    Args:
        config: Operational configuration
        embedding_model: Name of embedding model

    Returns:
        Configured StoreManager instance

    Raises:
        OSError: If Docker PostgreSQL startup fails
    """
    database_url = config.database_url

    # Check if we should auto-start Docker PostgreSQL
    # Note: database_url may already be worktree-specific from OperationalConfig.__post_init__
    from ragzoom.worktree_utils import (
        DEFAULT_DATABASE_NAME,
        DEFAULT_DATABASE_URL_TEMPLATE,
        get_worktree_database_name,
    )

    # Check if URL matches expected patterns (base or worktree-specific)
    expected_base_url = DEFAULT_DATABASE_URL_TEMPLATE.format(
        database_name=DEFAULT_DATABASE_NAME
    )
    expected_worktree_db_name = get_worktree_database_name()
    expected_worktree_url = DEFAULT_DATABASE_URL_TEMPLATE.format(
        database_name=expected_worktree_db_name
    )

    should_auto_start = (
        (database_url == expected_base_url or database_url == expected_worktree_url)
        and not os.getenv("RAGZOOM_DATABASE_URL")  # User didn't explicitly set URL
        and not os.getenv("RAGZOOM_NO_DOCKER")  # User didn't disable Docker
    )

    if should_auto_start:
        try:
            from ragzoom.docker_postgres import DockerPostgres

            docker_pg = DockerPostgres()

            # Use the expected database name for consistency
            if database_url == expected_worktree_url:
                # This is a worktree-specific database
                database_url = docker_pg.ensure_database_exists(
                    expected_worktree_db_name
                )
                logger.info(
                    f"✅ PostgreSQL ready with worktree database: {expected_worktree_db_name}"
                )
            else:
                # This is the base ragzoom database
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

    return StoreManager(config, embedding_model)


class StoreManager:
    """System-wide store manager that creates document-scoped stores and manages repositories.

    This class is responsible for:
    - Creating document-scoped stores via for_document()
    - Managing system-wide operations (multi-document)
    - Providing direct repository access for advanced usage
    - Database lifecycle management

    For document-specific operations, use store_manager.for_document(doc_id)
    For system-wide operations, use store_manager methods directly
    For advanced usage, access repositories via store_manager.nodes, etc.
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
                    openai_api_key=SecretStr(os.getenv("OPENAI_API_KEY", "test-key")),
                    database_url=temp_db_url,
                )

                # Create the store manager
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

    # Document-scoped store factory - primary API
    def for_document(self, document_id: str | None) -> DocumentStore:
        """Get a document-scoped store that prevents cross-document contamination.

        This is the primary API for all document-specific operations.
        All queries are automatically filtered to the specified document.

        Args:
            document_id: Document ID to scope operations to

        Returns:
            DocumentStore with automatic document filtering
        """
        return DocumentStore(
            document_id=document_id,
            node_repo=self.node_repo,
            search_service=self.search_service,
            tree_navigator=self.tree_navigator,
        )

    # Multi-document management operations
    def list_documents(self) -> list[Document]:
        """List all documents in the system."""
        with self.SessionLocal() as session:
            return session.query(Document).all()

    def get_document_by_path(self, file_path: str) -> Document | None:
        """Get a document by file path."""
        return self.doc_repo.get_document_by_path(file_path)

    def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        return self.doc_repo.get_document_by_id(document_id)

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

    def clear_document(self, document_id: str, *, session=None) -> int:
        """Clear all data for a document, including orphaned nodes and document record."""
        return self.doc_repo.clear_document(document_id, session=session)

    def clear_all_documents(self) -> int:
        """Clear all documents and nodes from the system."""
        total_cleared = 0
        with self.SessionLocal() as session:
            documents = session.query(Document).all()
            for doc in documents:
                total_cleared += self.clear_document(doc.id, session=session)
        return total_cleared

    # System-wide statistics and operations
    def get_total_node_count(self) -> int:
        """Get total number of nodes across all documents."""
        with self.SessionLocal() as session:
            return session.query(TreeNode).count()

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get all pinned nodes across all documents."""
        return self.node_repo.get_pinned_nodes(depth_max)

    def pin_node(self, node_id: str) -> None:
        """Pin a node if it's within allowed depth.

        Raises:
            NodeNotFoundError: If the node does not exist
            InvalidOperationError: If the node is too deep or already pinned
        """
        node = self.node_repo.get_node(node_id)
        if not node:
            raise NodeNotFoundError(node_id)

        node_depth = self.tree_navigator.get_node_depth(node_id)
        if node_depth > self.PIN_DEPTH_MAX:
            raise InvalidOperationError(
                "pin_node",
                f"Node {node_id} is at depth {node_depth}, which exceeds maximum pin depth {self.PIN_DEPTH_MAX}",
            )

        # Check if already pinned
        if node.is_pinned == 1:
            raise InvalidOperationError("pin_node", f"Node {node_id} is already pinned")

        self.node_repo.pin_node(node_id)

    # Legacy compatibility - expose repositories for advanced usage
    @property
    def nodes(self) -> NodeRepository:
        """Access to node repository for advanced operations."""
        return self.node_repo

    @property
    def documents(self) -> DocumentRepository:
        """Access to document repository for advanced operations."""
        return self.doc_repo

    @property
    def search(self) -> SearchService:
        """Access to search service for advanced operations."""
        return self.search_service

    @property
    def tree(self) -> TreeNavigator:
        """Access to tree navigator for advanced operations."""
        return self.tree_navigator

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

    # Database validation methods
    def _validate_embedding_dimension(
        self, embedding: list[float] | NDArray[np.float64]
    ) -> None:
        """Validate that embedding has correct dimension."""
        self.db_manager.validate_embedding_dimension(embedding)

    def _get_expected_embedding_dimension(self) -> int | None:
        """Get expected embedding dimension from existing embeddings."""
        return self.db_manager._get_expected_embedding_dimension()

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


# Alias for backward compatibility
Store = StoreManager
