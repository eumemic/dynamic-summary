"""Simplified mock Store implementation for fast testing."""

import hashlib
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
from numpy.typing import NDArray

from ragzoom.interfaces import StoreInterface
from ragzoom.models import Document, TreeNode


class MockTreeNode(SimpleNamespace):
    """Mock TreeNode that extends SimpleNamespace with helper methods."""

    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return self.height == 0

    def is_root(self) -> bool:
        """Check if this node is the root node."""
        return self.parent_id is None

    def is_left_child(self) -> bool:
        """Check if this node is a left child."""
        return self.path.endswith("0") if self.path else False

    def is_right_child(self) -> bool:
        """Check if this node is a right child."""
        return self.path.endswith("1") if self.path else False


class SimpleMockStore(StoreInterface):
    """Lightweight mock of Store for unit testing.

    This mock provides fast, in-memory implementations of the most commonly
    used Store methods. It maintains consistency for tree operations and
    basic state management without the overhead of real database I/O.

    Implements StoreInterface for type safety and compatibility.
    """

    PIN_DEPTH_MAX = 2  # Match Store class constant

    def __init__(self, config=None):
        self.config = config
        self._nodes: dict[str, SimpleNamespace] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.document_nodes: dict[str, set[str]] = defaultdict(set)
        self._documents: dict[str, SimpleNamespace] = {}

        # Transaction state tracking
        self._active_transaction = False
        # Transaction snapshot for rollback simulation
        self._transaction_snapshot: dict[str, any] | None = None

        # State tracking
        self.pinned_nodes: set[str] = set()
        self.mock_scores: dict[str, float] = {}

        # Track current document_id for set_metadata calls
        self.document_id: str | None = None

        # Cache simulation
        self._node_cache = OrderedDict()
        self._cache_order = deque(maxlen=1000)

        # Track expected embedding dimension
        self._expected_embedding_dim = None

        # Add node_repo attribute for CLI pin command compatibility
        self.node_repo = self  # SimpleMockStore acts as its own node repository

        # SessionLocal mock with filter_by support
        mock_session = MagicMock()
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)

        def mock_query(model_class):
            query_mock = MagicMock()

            # Determine model type
            is_treenode = (
                hasattr(model_class, "__name__") and "TreeNode" in model_class.__name__
            )
            is_document = (
                hasattr(model_class, "__name__") and "Document" in model_class.__name__
            )

            # Basic count() method
            if is_treenode:
                query_mock.count.return_value = len(self._nodes)
            elif is_document:
                query_mock.count.return_value = len(self._documents)
            else:
                query_mock.count.return_value = 0

            # Add filter_by support
            def mock_filter_by(**kwargs):
                filtered_query = MagicMock()

                if is_treenode:
                    # Filter nodes based on criteria
                    filtered_nodes = []
                    for node in self._nodes.values():
                        match = True
                        for key, value in kwargs.items():
                            node_value = getattr(node, key, None)
                            if node_value != value:
                                match = False
                                break
                        if match:
                            filtered_nodes.append(node)

                    filtered_query.count.return_value = len(filtered_nodes)
                    filtered_query.all.return_value = filtered_nodes
                    filtered_query.first.return_value = (
                        filtered_nodes[0] if filtered_nodes else None
                    )
                    filtered_query.delete.return_value = len(filtered_nodes)

                    # Add support for chained .filter() calls
                    def mock_filter(*args):
                        # Filter for leaf nodes (left_child_id.is_(None), right_child_id.is_(None))
                        further_filtered = []
                        for node in filtered_nodes:
                            # Check if it's a leaf node
                            if (
                                hasattr(node, "left_child_id")
                                and hasattr(node, "right_child_id")
                                and node.left_child_id is None
                                and node.right_child_id is None
                            ):
                                further_filtered.append(node)

                        result_query = MagicMock()
                        result_query.all.return_value = further_filtered
                        result_query.first.return_value = (
                            further_filtered[0] if further_filtered else None
                        )
                        result_query.count.return_value = len(further_filtered)
                        return result_query

                    filtered_query.filter = mock_filter

                elif is_document:
                    # Filter documents based on criteria
                    filtered_docs = []
                    for doc in self._documents.values():
                        match = True
                        for key, value in kwargs.items():
                            doc_value = getattr(doc, key, None)
                            if doc_value != value:
                                match = False
                                break
                        if match:
                            filtered_docs.append(doc)

                    filtered_query.count.return_value = len(filtered_docs)
                    filtered_query.all.return_value = filtered_docs
                    filtered_query.first.return_value = (
                        filtered_docs[0] if filtered_docs else None
                    )
                    filtered_query.delete.return_value = len(filtered_docs)
                else:
                    # Unknown model type
                    filtered_query.count.return_value = 0
                    filtered_query.all.return_value = []
                    filtered_query.first.return_value = None
                    filtered_query.delete.return_value = 0

                return filtered_query

            query_mock.filter_by = mock_filter_by
            query_mock.all.return_value = (
                list(self._nodes.values())
                if is_treenode
                else list(self.documents.values())
            )

            return query_mock

        mock_session.query = mock_query
        self.SessionLocal = MagicMock(return_value=mock_session)

        # Add StoreManager-compatible properties
        mock_search = MagicMock()
        mock_search.search_similar = self.search_similar
        mock_search.compute_mmr_diverse_results = self.compute_mmr_diverse_results
        self.search = mock_search

        mock_tree = MagicMock()
        mock_tree.get_ancestors = self.get_ancestors
        mock_tree.is_leaf_node = self.is_leaf_node
        mock_tree.is_root_node = self.is_root_node
        mock_tree.get_node_depth = self.get_node_depth
        mock_tree.get_children = self.get_children
        self.tree = mock_tree

        mock_nodes = MagicMock()
        mock_nodes.get_node = self.get_node
        mock_nodes.get_nodes = self.get_nodes
        mock_nodes.get_nodes_by_paths = self.get_nodes_by_paths
        mock_nodes.update_node_access = self.update_node_access
        mock_nodes.add_nodes_batch = self.add_nodes_batch
        mock_nodes.update_parent_references_batch = self.update_parent_references_batch
        # Add get method that delegates to get_node for compatibility with Assembler
        mock_nodes.get = self.get_node
        self.nodes = mock_nodes

        # Create a documents property that acts like both dict and repository
        class DocumentsProperty:
            def __init__(self, mock_store):
                self._store = mock_store
                self.get_document_embedding_model = (
                    mock_store.get_document_embedding_model
                )

            def __getitem__(self, key):
                return self._store._documents[key]

            def __setitem__(self, key, value):
                self._store._documents[key] = value

            def __delitem__(self, key):
                del self._store._documents[key]

            def __contains__(self, key):
                return key in self._store._documents

            def __iter__(self):
                return iter(self._store._documents)

            def __len__(self):
                return len(self._store._documents)

            def values(self):
                return self._store._documents.values()

            def keys(self):
                return self._store._documents.keys()

            def items(self):
                return self._store._documents.items()

            def get(self, key, default=None):
                return self._store._documents.get(key, default)

        self.documents = DocumentsProperty(self)

    def for_document(self, document_id: str | None):
        """Create a mock document store for testing."""
        # Return a mock DocumentStore-like object that delegates to this mock store
        mock_doc_store = MagicMock()
        mock_doc_store.document_id = document_id

        # Create mock nodes, search, tree that filter by document_id
        mock_nodes = MagicMock()

        def get_node_fn(node_id):
            if document_id is None:
                return self.get_node(node_id)
            node = self.get_node(node_id)
            if node and node.document_id == document_id:
                return node
            return None

        mock_nodes.get = get_node_fn
        mock_nodes.get_node = get_node_fn  # Both get and get_node should work
        mock_nodes.get_many = lambda node_ids: self.get_nodes(node_ids)
        # Add get_nodes alias for get_many (used by Retriever and CoverageBuilder)
        mock_nodes.get_nodes = lambda node_ids: self.get_nodes(node_ids)
        # Add get_nodes_by_paths (used by CoverageBuilder for siblings)
        mock_nodes.get_nodes_by_paths = lambda paths: [
            n
            for n in self._nodes.values()
            if hasattr(n, "path") and n.path in paths and n.document_id == document_id
        ]
        # Add update_access method (used by CoverageBuilder)
        mock_nodes.update_access = lambda node_id: self.update_node_access(node_id)
        mock_nodes.get_all = lambda: [
            n
            for n in self._nodes.values()
            if getattr(n, "document_id", None) == document_id
        ]
        mock_nodes.get_all_paginated = (
            lambda *, page_size=1000: self.get_all_nodes_for_document_paginated(
                document_id, page_size=page_size
            )
        )
        mock_nodes.get_leaves = lambda: [
            n
            for n in self.get_leaf_nodes()
            if getattr(n, "document_id", None) == document_id
        ]
        mock_nodes.add = self.add_node
        mock_nodes.add_batch = self.add_nodes_batch
        mock_nodes.update_access = self.update_node_access
        mock_nodes.update_parent_references_batch = self.update_parent_references_batch

        mock_search = MagicMock()
        mock_search.similar = lambda embedding, n: self.search_similar(
            embedding, n, where={"document_id": document_id} if document_id else None
        )
        mock_search.mmr_diverse = self.compute_mmr_diverse_results

        mock_tree = MagicMock()
        mock_tree.get_children = self.get_children
        # Filter ancestors by document_id for proper isolation
        mock_tree.get_ancestors = lambda node_ids: [
            ancestor
            for ancestor in self.get_ancestors(node_ids)
            if getattr(ancestor, "document_id", None) == document_id
        ]
        mock_tree.get_root = lambda: self.get_root_node_for_document(document_id)
        mock_tree.get_depth = self.get_node_depth
        mock_tree.is_leaf = self.is_leaf_node
        mock_tree.is_root = self.is_root_node

        mock_doc_store.nodes = mock_nodes
        mock_doc_store.search = mock_search
        mock_doc_store.tree = mock_tree

        # Add DocumentStore methods needed by CoverageBuilder
        mock_doc_store.PIN_DEPTH_MAX = self.PIN_DEPTH_MAX
        mock_doc_store.get_pinned_nodes = lambda depth_max=None: [
            node
            for node in self.get_pinned_nodes(depth_max)
            if node.document_id == document_id
        ]

        # Add set_metadata method needed by IndexingService
        def _set_metadata(**kwargs):
            # Save the current document_id, set it temporarily for the method call
            old_doc_id = self.document_id
            self.document_id = document_id
            self.set_metadata(**kwargs)
            self.document_id = old_doc_id

        mock_doc_store.set_metadata = _set_metadata
        mock_doc_store.compute_content_hash = self.compute_content_hash
        mock_doc_store.session_local = self.SessionLocal
        # Add new methods for Phase 4 refactoring
        mock_doc_store.get_embedding_model = lambda: (
            self.get_document_embedding_model(document_id) if document_id else None
        )
        mock_doc_store.get_avg_leaf_tokens = lambda: (
            self._get_avg_leaf_tokens_for_document(document_id) if document_id else None
        )

        return mock_doc_store

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
        preceding_neighbor_id: str | None = None,
        following_neighbor_id: str | None = None,
        path: str = "",
        is_left_child: bool | None = None,
    ) -> TreeNode:
        """Add a node to the mock store."""
        # Convert embedding to list if needed
        if isinstance(embedding, np.ndarray):
            embedding = embedding.tolist()

        # Validate embedding dimension
        if self._expected_embedding_dim is None:
            self._expected_embedding_dim = len(embedding)
        elif len(embedding) != self._expected_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._expected_embedding_dim}, "
                f"got {len(embedding)}"
            )

        # Compute token_count if not provided
        if token_count == 0:
            try:
                from ragzoom.utils.tokenization import count_tokens

                token_count = count_tokens(text)
            except (ImportError, Exception):
                # Fallback: approximate tokens as words
                token_count = len(text.split())

        # Calculate path based on parent relationship if not explicitly provided
        if path == "" and parent_id:
            parent = self._nodes.get(parent_id)
            if parent:
                # Use explicit child position if provided
                if is_left_child is not None:
                    path = parent.path + ("0" if is_left_child else "1")
                else:
                    # Determine from parent's current child pointers
                    if parent.left_child_id == node_id:
                        path = parent.path + "0"
                    elif parent.right_child_id == node_id:
                        path = parent.path + "1"
                    # If neither matches, the relationship might not be established yet
                    # In that case, keep the empty path for now

        node = MockTreeNode(
            id=node_id,
            parent_id=parent_id,
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            span_start=span_start,
            span_end=span_end,
            text=text,
            document_id=document_id,
            token_count=token_count,
            height=height,
            is_pinned=0,
            last_accessed=None,
            access_count=0,
            created_at=None,
            preceding_neighbor_id=preceding_neighbor_id,
            following_neighbor_id=following_neighbor_id,
            path=path,  # Binary tree path
            embedding=list(embedding),  # Store embedding in node
        )

        self._nodes[node_id] = node
        self.embeddings[node_id] = list(embedding)

        if document_id:
            self.document_nodes[document_id].add(node_id)

        # Add to cache
        self._add_to_cache(node)

        return node

    def add_nodes_batch(
        self, nodes_data: list[dict[str, Any]], *, session=None
    ) -> list[TreeNode]:
        """Add multiple nodes in batch - mock implementation."""
        nodes = []
        for data in nodes_data:
            node = self.add_node(
                node_id=data["node_id"],
                text=data["text"],
                embedding=data["embedding"],
                span_start=data["span_start"],
                span_end=data["span_end"],
                parent_id=data.get("parent_id"),
                left_child_id=data.get("left_child_id"),
                right_child_id=data.get("right_child_id"),
                document_id=data.get("document_id"),
                token_count=data.get("token_count", 0),
                height=data.get("height", 0),
                preceding_neighbor_id=data.get("preceding_neighbor_id"),
                following_neighbor_id=data.get("following_neighbor_id"),
                path=data.get("path", ""),
            )
            nodes.append(node)
        return nodes

    def update_parent_references_batch(
        self, updates: list[tuple[str, str]], *, session=None
    ) -> None:
        """Update parent references in batch - mock implementation."""
        for node_id, parent_id in updates:
            if node_id in self._nodes:
                self._nodes[node_id].parent_id = parent_id
                # Invalidate cache for updated node
                if node_id in self._node_cache:
                    del self._node_cache[node_id]
                    if node_id in self._cache_order:
                        self._cache_order.remove(node_id)

    def get_node(self, node_id: str) -> TreeNode | None:
        """Get a node by ID."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            # Update access tracking
            node.access_count = getattr(node, "access_count", 0) + 1
            # Move to end of cache order (most recently used)
            if node_id in self._node_cache:
                self._move_to_cache_end(node_id)
            return node
        return None

    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]:
        """Get multiple nodes by their IDs."""
        return [
            node for node_id in node_ids if (node := self.get_node(node_id)) is not None
        ]

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node."""
        if node_id in self._nodes:
            node = self._nodes[node_id]
            node.access_count = getattr(node, "access_count", 0) + 1

    def get_children(self, node_id: str) -> tuple[TreeNode | None, TreeNode | None]:
        """Get left and right children of a node."""
        node = self.get_node(node_id)
        if not node:
            return None, None

        left = self.get_node(node.left_child_id) if node.left_child_id else None
        right = self.get_node(node.right_child_id) if node.right_child_id else None
        return left, right

    def get_leaf_nodes(self) -> list[TreeNode]:
        """Get all leaf nodes (nodes with no children)."""
        return [
            node
            for node in self._nodes.values()
            if not node.left_child_id and not node.right_child_id
        ]

    def get_root_node(self) -> TreeNode | None:
        """Get the root node (node with no parent)."""
        for node in self._nodes.values():
            if not node.parent_id:
                return node
        return None

    def get_root_node_for_document(self, document_id: str | None) -> TreeNode | None:
        """Get the root node for a specific document."""
        for node in self._nodes.values():
            if node.document_id == document_id and not node.parent_id:
                return node
        return None

    def get_ancestors(self, node_ids: list[str]) -> list[TreeNode]:
        """Get all ancestors of given nodes."""
        all_ancestors = set()
        current_level = set(node_ids)

        while current_level:
            next_level = set()
            for node_id in current_level:
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    if node.parent_id and node.parent_id not in all_ancestors:
                        all_ancestors.add(node.parent_id)
                        next_level.add(node.parent_id)
            current_level = next_level

        return [
            self._nodes[ancestor_id]
            for ancestor_id in all_ancestors
            if ancestor_id in self._nodes
        ]

    def get_nodes_by_paths(self, paths: list[str]) -> list[TreeNode]:
        """Get nodes by their path values."""
        return [
            node
            for node in self._nodes.values()
            if hasattr(node, "path") and node.path in paths
        ]

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node (distance from root)."""
        if node_id not in self._nodes:
            raise ValueError(f"Node {node_id} not found")

        depth = 0
        current_id = node_id
        visited = set()

        while current_id and current_id not in visited:
            visited.add(current_id)
            node = self._nodes[current_id]
            if not node.parent_id:
                break
            current_id = node.parent_id
            depth += 1

        return depth

    def is_leaf_node(self, node_id: str) -> bool:
        """Check if a node is a leaf (has no children)."""
        if node_id not in self._nodes:
            return False
        node = self._nodes[node_id]
        return not node.left_child_id and not node.right_child_id

    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is a root (has no parent)."""
        if node_id not in self._nodes:
            return False
        return not self._nodes[node_id].parent_id

    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]:
        """Get all nodes for a specific document."""
        return [
            node for node in self._nodes.values() if node.document_id == document_id
        ]

    def get_all_nodes_for_document_paginated(
        self, document_id: str | None, *, page_size: int = 1000
    ) -> list[list[TreeNode]]:
        """Get all nodes for a document in paginated batches for memory efficiency.

        Mock implementation that simulates the paginated behavior.
        """
        if page_size <= 0:
            raise ValueError("page_size must be positive")

        all_nodes = self.get_all_nodes_for_document(document_id)
        if not all_nodes:
            return []

        # Split into batches
        batches = []
        for i in range(0, len(all_nodes), page_size):
            batch = all_nodes[i : i + page_size]
            batches.append(batch)

        return batches

    def search_similar(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        n_results: int,
        where: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        """Search for similar nodes using cosine similarity."""
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        results = []
        for node_id, node_embedding in self.embeddings.items():
            # Simple cosine similarity calculation
            dot_product = sum(a * b for a, b in zip(query_embedding, node_embedding))
            norm_a = sum(a * a for a in query_embedding) ** 0.5
            norm_b = sum(b * b for b in node_embedding) ** 0.5

            if norm_a > 0 and norm_b > 0:
                similarity = dot_product / (norm_a * norm_b)
            else:
                similarity = 0.0

            # Check if we have a mock score override
            if node_id in self.mock_scores:
                similarity = self.mock_scores[node_id]

            node = self._nodes[node_id]
            metadata = {
                "span_start": node.span_start,
                "span_end": node.span_end,
                "parent_id": node.parent_id or "",
                "document_id": node.document_id or "",
                "is_leaf": 1 if self.is_leaf_node(node_id) else 0,
            }

            # Apply where filter if provided
            if where:
                if "document_id" in where and node.document_id != where["document_id"]:
                    continue

            results.append((node_id, similarity, metadata))

        # Sort by similarity (descending) and take top n_results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:n_results]

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float] | NDArray[np.float64],
        candidates: list[tuple[str, float, dict[str, Any]]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Apply MMR (Maximal Marginal Relevance) to get diverse results."""
        if not candidates or k <= 0:
            return []

        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()

        selected = []
        remaining = candidates.copy()

        for _ in range(min(k, len(candidates))):
            if not remaining:
                break

            best_score = -float("inf")
            best_idx = 0

            for i, (node_id, relevance, _) in enumerate(remaining):
                # Calculate diversity penalty
                max_similarity = 0.0
                if selected:
                    node_embedding = self.embeddings.get(node_id, [])
                    for selected_id in selected:
                        selected_embedding = self.embeddings.get(selected_id, [])
                        if node_embedding and selected_embedding:
                            # Simple cosine similarity
                            dot_product = sum(
                                a * b
                                for a, b in zip(node_embedding, selected_embedding)
                            )
                            norm_a = sum(a * a for a in node_embedding) ** 0.5
                            norm_b = sum(b * b for b in selected_embedding) ** 0.5
                            if norm_a > 0 and norm_b > 0:
                                similarity = dot_product / (norm_a * norm_b)
                                max_similarity = max(max_similarity, similarity)

                # MMR score: λ * relevance - (1 - λ) * max_similarity
                mmr_score = (
                    lambda_param * relevance - (1 - lambda_param) * max_similarity
                )

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i

            # Select the best candidate
            selected_node_id = remaining[best_idx][0]
            selected.append(selected_node_id)
            remaining.pop(best_idx)

        return selected

    def get_document_by_path(self, file_path: str) -> Document | None:
        """Get a document by file path."""
        for doc in self.documents.values():
            if doc.file_path == file_path:
                return doc
        return None

    def get_document_by_id(self, document_id: str) -> Document | None:
        """Get a document by ID."""
        return self.documents.get(document_id)

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
    ):
        """Mock add document and return a DocumentStore for it."""
        from datetime import datetime

        doc = SimpleNamespace(
            id=document_id,
            file_path=file_path,
            content_hash=content_hash,
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            summary_model=summary_model,
            created_at=datetime.now(),
            indexed_at=datetime.now(),  # Add missing attribute
        )
        self._documents[document_id] = doc
        return self.for_document(document_id)

    def update_parent_reference(self, node_id: str, parent_id: str) -> None:
        """Update a node's parent reference and invalidate cache.

        Args:
            node_id: ID of the node to update
            parent_id: New parent ID
        """
        if node_id in self._nodes:
            self._nodes[node_id].parent_id = parent_id
            # In mock store, we don't have a real cache to invalidate
            # but we maintain consistency with the real implementation

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Compute SHA256 hash of content."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def set_metadata(
        self,
        file_path: str | None = None,
        content_hash: str | None = None,
        chunk_count: int = 0,
        embedding_model: str | None = None,
        summary_model: str | None = None,
    ) -> None:
        """Mock method for setting document metadata.

        Creates or updates a Document record in the mock store.
        """
        from datetime import datetime

        if not self.document_id:
            return  # Can't set metadata without a document ID

        # Create or update the document record
        if self.document_id not in self._documents:
            # Create new document
            doc = SimpleNamespace(
                id=self.document_id,
                file_path=file_path,
                content_hash=content_hash,
                chunk_count=chunk_count,
                embedding_model=embedding_model,
                summary_model=summary_model,
                indexed_at=datetime.utcnow(),
            )
            self._documents[self.document_id] = doc
        else:
            # Update existing document
            doc = self._documents[self.document_id]
            if file_path is not None:
                doc.file_path = file_path
            if content_hash is not None:
                doc.content_hash = content_hash
            if chunk_count > 0:
                doc.chunk_count = chunk_count
            if embedding_model is not None:
                doc.embedding_model = embedding_model
            if summary_model is not None:
                doc.summary_model = summary_model

    def pin_node(self, node_id: str) -> None:
        """Pin a node."""
        if node_id in self._nodes:
            self.pinned_nodes.add(node_id)
            self._nodes[node_id].is_pinned = 1

    def set_mock_scores(self, scores: dict[str, float]):
        """Set mock similarity scores for testing."""
        self.mock_scores = scores

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]:
        """Get all pinned nodes up to optional max depth."""
        pinned = []
        for node_id in self.pinned_nodes:
            if node_id in self._nodes:
                if depth_max is None or self.get_node_depth(node_id) <= depth_max:
                    pinned.append(self._nodes[node_id])
        return pinned

    def get_document_embedding_model(self, document_id: str) -> str | None:
        """Get the embedding model used for a specific document."""
        doc = self.get_document_by_id(document_id)
        if isinstance(doc, dict):
            return doc.get("embedding_model")
        return doc.embedding_model if doc else None

    def _get_avg_leaf_tokens_for_document(self, document_id: str) -> int | None:
        """Get average token count for leaf nodes in this document."""
        leaf_nodes = [
            node
            for node in self._nodes.values()
            if node.document_id == document_id and self.is_leaf_node(node.id)
        ]
        if not leaf_nodes:
            return None

        total_tokens = sum(node.token_count for node in leaf_nodes)
        return total_tokens // len(leaf_nodes) if leaf_nodes else None

    def delete_document_nodes(self, document_id: str, *, session=None) -> int:
        """Delete all nodes for a document."""
        if document_id not in self.document_nodes:
            return 0

        node_ids = list(self.document_nodes[document_id])
        for node_id in node_ids:
            if node_id in self._nodes:
                del self._nodes[node_id]
            if node_id in self.embeddings:
                del self.embeddings[node_id]
            if node_id in self.pinned_nodes:
                self.pinned_nodes.remove(node_id)

        del self.document_nodes[document_id]
        return len(node_ids)

    def get_document_token_stats(self, document_id: str) -> dict[str, float | int]:
        """Get token statistics for a document."""
        nodes = self.get_all_nodes_for_document(document_id)
        if not nodes:
            return {"total_tokens": 0, "node_count": 0, "avg_tokens": 0.0}

        total_tokens = sum(node.token_count for node in nodes)
        node_count = len(nodes)
        avg_tokens = total_tokens / node_count if node_count > 0 else 0.0

        return {
            "total_tokens": total_tokens,
            "node_count": node_count,
            "avg_tokens": avg_tokens,
        }

    def clear_document(self, document_id: str, *, session=None) -> int:
        """Clear all data for a document, including orphaned nodes and document record."""
        deleted_count = self.delete_document_nodes(document_id, session=session)
        if document_id in self.documents:
            del self._documents[document_id]
        return deleted_count

    @contextmanager
    def transaction(self):
        """Mock transaction context manager with rollback simulation.

        Simulates transaction behavior by:
        - Taking a snapshot of current state at transaction start
        - Restoring snapshot on exception to simulate rollback
        - Preventing nested transactions like real Store
        """
        if self._active_transaction:
            raise RuntimeError(
                "Nested transactions are not supported. "
                "Please use the same session for all operations within a transaction."
            )

        # Take snapshot of current state
        self._active_transaction = True
        self._transaction_snapshot = {
            "documents": dict(self._documents),
            "nodes": dict(self._nodes),
            "embeddings": dict(self.embeddings),
            "document_nodes": {k: set(v) for k, v in self.document_nodes.items()},
        }

        # Create a mock session object for compatibility
        mock_session = MagicMock()

        try:
            yield mock_session
            # Transaction successful - clear snapshot
            self._transaction_snapshot = None
        except Exception:
            # Rollback - restore from snapshot
            if self._transaction_snapshot:
                self._documents = self._transaction_snapshot["documents"]
                self._nodes = self._transaction_snapshot["nodes"]
                self.embeddings = self._transaction_snapshot["embeddings"]
                self.document_nodes = self._transaction_snapshot["document_nodes"]
                self._transaction_snapshot = None
            raise
        finally:
            self._active_transaction = False

    def close(self) -> None:
        """Close database connections and cleanup resources."""
        pass  # No-op for mock

    @property
    def node_cache(self):
        """Access to node cache for backward compatibility."""
        return self._node_cache

    @property
    def cache_order(self):
        """Access to cache order for backward compatibility."""
        return self._cache_order

    def update_node_paths_from_tree_structure(self) -> None:
        """Update node paths based on the current tree structure.

        This method should be called after tree construction is complete to ensure
        all nodes have correct paths assigned based on their parent-child relationships.
        """
        # Find root nodes (nodes with no parent)
        root_nodes = [node for node in self._nodes.values() if node.is_root()]

        # Update paths starting from root nodes
        visited = set()
        for root in root_nodes:
            self._update_node_path_recursive(root, "", visited)

    def _update_node_path_recursive(self, node, path: str, visited: set[str]) -> None:
        """Recursively update node paths in the tree.

        Args:
            node: Current node to update
            path: Path to assign to this node
            visited: Set of visited node IDs to prevent infinite loops
        """
        if node.id in visited:
            return
        visited.add(node.id)

        # Update this node's path
        node.path = path

        # Update children
        if node.left_child_id and node.left_child_id in self._nodes:
            left_child = self._nodes[node.left_child_id]
            self._update_node_path_recursive(left_child, path + "0", visited)

        if node.right_child_id and node.right_child_id in self._nodes:
            right_child = self._nodes[node.right_child_id]
            self._update_node_path_recursive(right_child, path + "1", visited)

    def _add_to_cache(self, node) -> None:
        """Add a node to the cache."""
        if len(self._node_cache) >= 1000:
            # Remove oldest item
            oldest_key = next(iter(self._node_cache))
            del self._node_cache[oldest_key]

        self._node_cache[node.id] = node
        if node.id in self._cache_order:
            self._cache_order.remove(node.id)
        self._cache_order.append(node.id)

    def _move_to_cache_end(self, node_id: str) -> None:
        """Move a node to the end of cache order (most recently used)."""
        if node_id in self._cache_order:
            self._cache_order.remove(node_id)
        self._cache_order.append(node_id)
