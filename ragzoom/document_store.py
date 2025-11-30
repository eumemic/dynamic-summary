"""Document-scoped store that prevents cross-document contamination."""

import logging
from collections.abc import Generator, Sequence
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING, cast

import numpy as np
from numpy.typing import NDArray
from sqlalchemy.orm import Session

from ragzoom.contracts.document_repository import DocumentRepository
from ragzoom.contracts.node_repository import (
    NodeDataDict,
)
from ragzoom.contracts.node_repository import (
    NodeRepository as NodeRepositoryProtocol,
)
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.error_handling import handle_graceful_error
from ragzoom.services.tree_navigator import TreeNavigator

if TYPE_CHECKING:
    from ragzoom.models import Document

logger = logging.getLogger(__name__)


class DocumentNodeRepository:
    """Node repository automatically scoped to a specific document."""

    def __init__(self, document_id: str | None, node_repo: NodeRepositoryProtocol):
        self.document_id = document_id
        self._repo = node_repo

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
        token_count: int = 0,
        height: int = 0,
        is_left_child: bool | None = None,
        level_index: int = 0,
    ) -> TreeNode:
        """Add a node scoped to this document."""
        # Ensure embedding type matches repository protocol
        emb: list[float]
        if isinstance(embedding, np.ndarray):
            emb = [float(x) for x in embedding.tolist()]
        else:
            emb = [float(x) for x in embedding]

        return self._repo.add_node(
            node_id=node_id,
            text=text,
            embedding=emb,
            span_start=span_start,
            span_end=span_end,
            parent_id=parent_id,
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            document_id=self.document_id,
            token_count=token_count,
            height=height,
            is_left_child=is_left_child,
            level_index=level_index,
        )

    def add_batch(
        self,
        nodes_data: list[NodeDataDict],
        *,
        session: Session | None = None,
    ) -> list[TreeNode]:
        """Add multiple nodes to this document in batch."""
        # Ensure all nodes have the document_id set
        processed_data: list[NodeDataDict] = []
        for node_data in nodes_data:
            processed_node: NodeDataDict = {
                **node_data,
                "document_id": self.document_id,
            }
            processed_data.append(processed_node)
        return self._repo.add_nodes_batch(processed_data, session=session)

    # jscpd:ignore-start - wrappers mirror repository signatures for document scoping
    def upsert_nodes_batch(
        self,
        nodes_data: list[NodeDataDict],
        *,
        session: Session | None = None,
    ) -> list[TreeNode]:
        """Upsert multiple nodes while enforcing document scope."""
        processed: list[NodeDataDict] = []
        for node_data in nodes_data:
            processed_node: NodeDataDict = {
                **node_data,
                "document_id": self.document_id,
            }
            processed.append(processed_node)
        return self._repo.upsert_nodes_batch(processed, session=session)

    def delete_nodes(
        self,
        node_ids: Sequence[str],
        *,
        session: Session | None = None,
    ) -> None:
        """Delete a set of nodes while enforcing document scope."""

        deleter = getattr(self._repo, "delete_nodes", None)
        if not callable(deleter):
            raise NotImplementedError("Underlying repository does not support deletion")

        scoped: list[str] = []
        for node_id in node_ids:
            if not node_id:
                continue
            node = self.get(node_id)
            if node is None:
                continue
            scoped.append(node_id)

        if not scoped:
            return
        deleter(scoped, session=session)

    def get(self, node_id: str) -> TreeNode | None:
        """Get a node by ID, ensuring it belongs to this document."""
        node = self._repo.get_node(node_id)
        if node and node.document_id == self.document_id:
            return node
        return None

    # Alias for compatibility with repository naming
    def get_node(self, node_id: str) -> TreeNode | None:
        """Alias of get() to match NodeRepository interface methods."""
        return self.get(node_id)

    # Backward-compatible alias to match NodeRepository interface
    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by IDs, filtered to this document only."""
        nodes = self._repo.get_nodes(node_ids)
        return [node for node in nodes if node.document_id == self.document_id]

    def get_by_height_and_level(
        self, *, height: int, level_index: int
    ) -> TreeNode | None:
        """Lookup a node at a specific height/level within this document."""

        getter = getattr(self._repo, "get_node_by_height_and_level", None)
        if not callable(getter):
            return None
        return cast(
            TreeNode | None,
            getter(self.document_id, height, level_index),
        )

    def get_by_height_levels(
        self, coordinates: Sequence[tuple[int, int]]
    ) -> list[TreeNode]:
        """Bulk lookup by coordinate pairs within this document."""

        getter = getattr(self._repo, "get_nodes_by_height_levels", None)
        if not callable(getter):
            return []
        return cast(
            list[TreeNode],
            getter(self.document_id, coordinates),
        )

    def get_root_nodes(self, document_id: str | None = None) -> list[TreeNode]:
        """Get root nodes scoped to the provided or default document."""

        target_doc = document_id or self.document_id
        nodes = self._repo.get_root_nodes(target_doc)
        if target_doc is None:
            return nodes
        return [node for node in nodes if node.document_id == target_doc]

    def get_parentless_nodes(self) -> list[TreeNode]:
        """Return nodes without parents for this document."""

        getter = getattr(self._repo, "get_parentless_nodes_for_document", None)
        if not callable(getter):
            raise NotImplementedError(
                "Underlying repository does not support parentless node queries"
            )
        return list(getter(self.document_id))

    def get_ready_left_children(self) -> list[str]:
        getter = getattr(self._repo, "get_ready_left_children", None)
        if not callable(getter):
            raise NotImplementedError(
                "Underlying repository does not support ready left child queries"
            )
        return list(getter(self.document_id))

    def get_rightmost_leaf_for_document(
        self, document_id: str | None
    ) -> TreeNode | None:
        target_doc = document_id or self.document_id
        getter = getattr(self._repo, "get_rightmost_leaf_for_document", None)
        if not callable(getter):
            raise NotImplementedError(
                "Underlying repository does not support rightmost leaf lookup"
            )
        node = getter(target_doc)
        if node and (target_doc is None or node.document_id == target_doc):
            return cast(TreeNode, node)
        return None

    def get_many(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes, filtering to this document only."""
        nodes = self._repo.get_nodes(node_ids)
        return [node for node in nodes if node.document_id == self.document_id]

    def get_all(self) -> list[TreeNode]:
        """Get all nodes for this document."""
        return self._repo.get_all_nodes_for_document(self.document_id)

    def get_all_paginated(self, *, page_size: int = 1000) -> list[list[TreeNode]]:
        """Get all nodes for this document in paginated batches."""
        return self._repo.get_all_nodes_for_document_paginated(
            self.document_id, page_size=page_size
        )

    def get_in_span(
        self,
        span_start: int,
        span_end: int,
        *,
        limit: int,
        min_height: int | None = None,
    ) -> tuple[list[TreeNode], int]:
        """Return nodes overlapping the requested span ordered for visualization."""
        getter = getattr(self._repo, "get_nodes_overlapping_span", None)
        if not callable(getter):
            raise NotImplementedError(
                "Underlying repository does not support span queries"
            )
        if self.document_id is None:
            raise ValueError(
                "Span queries require a document scope. "
                "Ensure DocumentStore.for_document(<document_id>) is used."
            )
        nodes, total = getter(
            self.document_id,
            int(span_start),
            int(span_end),
            limit=int(limit),
            min_height=None if min_height is None else int(min_height),
        )
        scoped = [node for node in nodes if node.document_id == self.document_id]
        return scoped, total

    def count(self) -> int:
        """Get count of nodes for this document efficiently."""
        counter = getattr(self._repo, "count_nodes_for_document", None)
        if callable(counter):
            return int(counter(self.document_id))
        # Fallback: materialize and count (less efficient)
        return len(self._repo.get_all_nodes_for_document(self.document_id))

    def leaf_count(self) -> int:
        """Get count of leaf nodes for this document efficiently."""
        counter = getattr(self._repo, "count_leaves_for_document", None)
        if callable(counter):
            return int(counter(self.document_id))
        return len(self.get_leaves())

    def max_height(self) -> int:
        """Get maximum height for nodes in this document efficiently."""
        getter = getattr(self._repo, "max_height_for_document", None)
        if callable(getter):
            return int(getter(self.document_id))
        nodes = self.get_all()
        return max((n.height for n in nodes), default=0)

    def pinned_count(self) -> int:
        """Get count of pinned nodes for this document efficiently."""
        counter = getattr(self._repo, "count_pinned_for_document", None)
        if callable(counter):
            return int(counter(self.document_id))
        return len(self._repo.get_pinned_nodes(None))

    def get_leaves(self) -> list[TreeNode]:
        """Get all leaf nodes for this document."""
        # Prefer document-scoped repository method when available to avoid full-table scans
        get_scoped = getattr(self._repo, "get_leaf_nodes_for_document", None)
        if callable(get_scoped):
            from typing import cast as _cast

            return _cast(list[TreeNode], get_scoped(self.document_id))
        all_leaves = self._repo.get_leaf_nodes()
        return [node for node in all_leaves if node.document_id == self.document_id]

    def get_recent_leaves_within_budget(self, token_budget: int) -> list[TreeNode]:
        """Get most recent leaves that fit within token budget.

        Returns leaves ordered by span_start (document order).
        Uses efficient SQL window function to avoid loading all leaves.
        """
        return self._repo.get_recent_leaves_within_budget(
            self.document_id, token_budget
        )

    def update_parent_references_batch(
        self,
        updates: Sequence[tuple[str, str | None]],
        *,
        session: Session | None = None,
    ) -> None:
        """Update parent references for nodes in this document."""
        # Note: We trust that the caller is only updating nodes from this document
        # as this is typically called during tree construction where document consistency is maintained
        self._repo.update_parent_references_batch(updates, session=session)

    def update_neighbors_batch(
        self,
        updates: list[tuple[str, str | None, str | None]],
        *,
        session: Session | None = None,
    ) -> None:
        updater = getattr(self._repo, "update_neighbors_batch", None)
        if not callable(updater):
            raise NotImplementedError(
                "Underlying repository does not support neighbor updates"
            )
        updater(updates, session=session)

    # jscpd:ignore-end


# Legacy DocumentSearchService removed; retrieval uses VectorIndex directly


class DocumentTreeNavigator:
    """Tree navigation automatically scoped to a specific document."""

    def __init__(self, document_id: str | None, tree_navigator: TreeNavigator):
        self.document_id = document_id
        self._navigator = tree_navigator

    def clear_depth_cache(self, node_ids: list[str]) -> None:
        self._navigator.clear_depth_cache(node_ids)

    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        """Get children of a node, verifying document scope."""
        # First verify the parent node belongs to this document
        parent = self._navigator.node_repo.get_node(node_id)
        if not parent or parent.document_id != self.document_id:
            return None, None

        return self._navigator.get_children(node_id)

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get ancestors of nodes within this document."""
        # Filter input nodes to this document first
        valid_nodes: list[str] = []
        for node_id in node_ids:
            node = self._navigator.node_repo.get_node(node_id)
            if node and node.document_id == self.document_id:
                valid_nodes.append(node_id)

        if not valid_nodes:
            return []

        ancestors = self._navigator.get_ancestors(valid_nodes)
        return [node for node in ancestors if node.document_id == self.document_id]

    def get_root(self) -> TreeNode | None:
        """Get the root node for this document."""
        return self._navigator.get_root_node_for_document(self.document_id)

    def get_depth(self, node_id: str) -> int:
        """Get depth of a node, verifying it belongs to this document."""
        node = self._navigator.node_repo.get_node(node_id)
        if not node or node.document_id != self.document_id:
            raise ValueError(f"Node {node_id} not found in document {self.document_id}")

        return self._navigator.get_node_depth(node_id)

    def is_leaf(self, node_id: str) -> bool:
        """Check if node is a leaf, verifying document scope."""
        node = self._navigator.node_repo.get_node(node_id)
        if not node or node.document_id != self.document_id:
            return False

        return self._navigator.is_leaf_node(node_id)

    def is_root(self, node_id: str) -> bool:
        """Check if node is root, verifying document scope."""
        node = self._navigator.node_repo.get_node(node_id)
        if not node or node.document_id != self.document_id:
            return False

        return self._navigator.is_root_node(node_id)

    def get_sibling(self, node_id: str) -> TreeNode | None:
        """Get sibling of a node within this document."""

        sibling = self._navigator.get_sibling_node(node_id)
        if sibling and sibling.document_id == self.document_id:
            return sibling
        return None

    def get_preceding_neighbor(self, node_id: str) -> TreeNode | None:
        """Get preceding neighbor of a node within this document."""

        neighbor = self._navigator.get_preceding_neighbor(node_id)
        if neighbor and neighbor.document_id == self.document_id:
            return neighbor
        return None

    def get_following_neighbor(self, node_id: str) -> TreeNode | None:
        """Get following neighbor of a node within this document."""

        neighbor = self._navigator.get_following_neighbor(node_id)
        if neighbor and neighbor.document_id == self.document_id:
            return neighbor
        return None


class DocumentStore:
    """Store scoped to a single document - prevents cross-contamination."""

    # Class constants from StoreManager (for compatibility)
    PIN_DEPTH_MAX = 2  # Maximum depth for pinned nodes

    def __init__(
        self,
        document_id: str | None,
        node_repo: NodeRepositoryProtocol,
        tree_navigator: TreeNavigator,
        doc_repo: DocumentRepository,
    ):
        """Initialize document-scoped store.

        Args:
            document_id: Document ID to scope all operations to
            node_repo: Node repository to wrap
            tree_navigator: Tree navigator to wrap
            doc_repo: Document repository for metadata access
        """
        self.document_id = document_id
        self._node_repo = node_repo  # Keep reference for pinned node operations
        self._doc_repo = doc_repo  # Keep reference for document metadata

        # Create document-scoped wrappers
        self.nodes = DocumentNodeRepository(document_id, node_repo)
        self.tree = DocumentTreeNavigator(document_id, tree_navigator)

        # Transaction state tracking (similar to StoreManager)
        self._active_transaction = False

    # jscpd:ignore-start - Transaction interface legitimately duplicates StoreManager method for consistency
    @contextmanager
    def transaction(self) -> Generator[Session, None, None]:
        """Context manager for transactional operations.

        Usage:
            with doc_store.transaction() as session:
                doc_store.nodes.add_batch(..., session=session)
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
        session_context = self._open_session()
        session = session_context.__enter__()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            self._active_transaction = False
            session_context.__exit__(None, None, None)

    def clear_document(
        self, document_id: str | None = None, *, session: Session | None = None
    ) -> int:
        """Delete all nodes for a document.

        Args:
            document_id: Document ID to clear (defaults to this store's document_id)
            session: Optional session for transactional operations

        Returns:
            Number of nodes deleted
        """
        target_doc_id = document_id or self.document_id
        if not target_doc_id:
            raise ValueError("Cannot clear document without a document_id")

        # Check if the underlying doc_repo has clear_document method
        if hasattr(self._doc_repo, "clear_document"):
            return self._doc_repo.clear_document(target_doc_id, session=session)
        else:
            # Fallback: use the clear method from this store
            if target_doc_id == self.document_id:
                return self.clear()
            else:
                raise ValueError(
                    "Cannot clear different document without doc_repo.clear_document support"
                )

    # jscpd:ignore-end

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get all pinned nodes for this document only."""
        # Prefer backend-optimized query if available to scope by document
        getter = getattr(self._node_repo, "get_pinned_nodes_for_document", None)
        if callable(getter) and self.document_id is not None:
            pinned: list[TreeNode] = list(getter(self.document_id, None))
        else:
            pinned = [
                node
                for node in self._node_repo.get_pinned_nodes(None)
                if node.document_id == self.document_id
            ]

        if depth_max is None:
            return pinned

        filtered: list[TreeNode] = []
        for node in pinned:
            try:
                depth = self.tree.get_depth(node.id)
            except Exception as exc:
                handle_graceful_error(
                    exc, f"Depth lookup failed for pinned node {node.id}", default=None
                )
                continue
            if depth_max is None or depth <= depth_max:
                filtered.append(node)
        return filtered

    def get_nodes_in_span(
        self,
        span_start: int,
        span_end: int,
        *,
        limit: int,
        min_height: int | None = None,
    ) -> tuple[list[TreeNode], int]:
        """Shortcut for DocumentNodeRepository.get_in_span."""
        return self.nodes.get_in_span(
            span_start,
            span_end,
            limit=limit,
            min_height=min_height,
        )

    def update_parent_reference(self, node_id: str, parent_id: str) -> None:
        """Update a node's parent reference and invalidate cache.

        Args:
            node_id: ID of the node to update
            parent_id: New parent ID
        """
        # Delegate to repository which handles cache invalidation
        self._node_repo.update_parent_references_batch([(node_id, parent_id)])

    def clear(self) -> int:
        """Delete all nodes for this document.

        Returns:
            Number of nodes deleted
        """
        if not self.document_id:
            raise ValueError("Cannot clear nodes without a document_id")

        # Delegate to the document repository's clear_document method
        # This ensures consistent behavior with StoreManager.clear_document()
        return self._doc_repo.clear_document(self.document_id)

    def _ensure_exists(self) -> None:
        """Ensure this document exists in the database.

        Raises:
            ValueError: If document_id is not set
            RuntimeError: If document does not exist in the database
        """
        if not self.document_id:
            raise ValueError("Cannot ensure document exists without a document_id")

        doc = self._doc_repo.get_document_by_id(self.document_id)
        if not doc:
            raise RuntimeError(
                f"Document '{self.document_id}' does not exist. "
                "Documents must be created before operations can be performed on them."
            )

    def get_metadata(self) -> "Document | None":
        """Get the document metadata record.

        Returns:
            Document record if it exists, None otherwise
        """
        if not self.document_id:
            return None

        from ragzoom.models import Document

        with self._open_session() as session:
            return session.query(Document).filter_by(id=self.document_id).first()

    # jscpd:ignore-start - Signature mirrors repository APIs for session support
    def set_metadata(
        self,
        file_path: str | None = None,
        embedding_model: str | None = None,
        summary_model: str | None = None,
        *,
        session: Session | None = None,
    ) -> None:
        """Set or update metadata for this document.

        Args:
            file_path: Optional file path
            embedding_model: Model used for embeddings
            summary_model: Model used for summaries
        """
        if not self.document_id:
            raise ValueError("Cannot set metadata without a document_id")

        from ragzoom.models import Document

        def _apply(session_obj: Session) -> None:
            doc = session_obj.query(Document).filter_by(id=self.document_id).first()

            if doc:
                if file_path is not None:
                    doc.file_path = file_path
                if embedding_model is not None:
                    doc.embedding_model = embedding_model
                if summary_model is not None:
                    doc.summary_model = summary_model
            else:
                doc = Document(
                    id=self.document_id,
                    file_path=file_path,
                    embedding_model=embedding_model,
                    summary_model=summary_model,
                )
                session_obj.add(doc)

        if session is not None:
            _apply(session)
            return

        with self._open_session() as session_ctx:
            _apply(session_ctx)
            session_ctx.commit()

    # jscpd:ignore-end

    def get_embedding_model(self) -> str | None:
        """Get the embedding model used for this document.

        Returns:
            Embedding model name if document exists, None otherwise
        """
        if not self.document_id:
            return None

        return self._doc_repo.get_document_embedding_model(self.document_id)

    def get_avg_leaf_tokens(self) -> int | None:
        """Get average token count for leaf nodes in this document.

        Returns:
            Average token count for leaves, or None if no data
        """
        if not self.document_id:
            logger.debug("No document_id provided for token statistics")
            return None

        leaves = self.nodes.get_leaves()
        if not leaves:
            logger.debug(
                f"No leaf nodes found for document {self.document_id} when calculating average tokens"
            )
            return None

        total_tokens = sum(int(n.token_count) for n in leaves)
        count = len(leaves)
        if count == 0:
            return None
        avg_tokens: int = total_tokens // count
        logger.debug(
            f"Calculated average leaf tokens: {avg_tokens} from {count} leaves for document {self.document_id}"
        )
        return avg_tokens

    # Internal helper to obtain a SQLAlchemy session from either backend type
    def _open_session(self) -> AbstractContextManager[Session]:
        session_local = getattr(self._node_repo, "SessionLocal", None)
        if callable(session_local):
            return session_local()  # type: ignore[no-any-return]
        db_manager = getattr(self._node_repo, "db_manager", None)
        if db_manager is not None and hasattr(db_manager, "SessionLocal"):
            return db_manager.SessionLocal()  # type: ignore[no-any-return]
        raise RuntimeError("No session factory available on node repository")
