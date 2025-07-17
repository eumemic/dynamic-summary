"""Mock Store implementation for fast testing."""

import hashlib
from collections import OrderedDict, defaultdict, deque
from types import SimpleNamespace
from typing import Optional
from unittest.mock import MagicMock


class SimpleMockStore:
    """Lightweight mock of Store for unit testing.

    This mock provides fast, in-memory implementations of the most commonly
    used Store methods. It maintains consistency for tree operations and
    basic state management without the overhead of real database I/O.
    """

    def __init__(self, config=None):
        self.config = config
        self.nodes: dict[str, SimpleNamespace] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.document_nodes: dict[str, set[str]] = defaultdict(set)
        self.documents: dict[str, SimpleNamespace] = {}

        # State tracking
        self.dirty_nodes: set[str] = set()
        self.pinned_nodes: set[str] = set()
        self.mock_scores: dict[str, float] = {}

        # Cache simulation
        self.node_cache = OrderedDict()
        self.cache_order = deque(maxlen=1000)

        # Mock SessionLocal for tests that use direct DB access
        self._setup_session_mock()

        # Track expected embedding dimension
        self._expected_embedding_dim = None

    def _setup_session_mock(self):
        """Setup mock session for compatibility with tests using SessionLocal."""
        mock_session = MagicMock()
        mock_query = MagicMock()

        # Make query chainable
        mock_query.filter_by.return_value = mock_query
        mock_query.filter.return_value = mock_query
        mock_query.all.return_value = []
        mock_query.first.return_value = None
        mock_query.count.return_value = 0

        mock_session.query.return_value = mock_query
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)

        self.SessionLocal = MagicMock(return_value=mock_session)
        self._mock_session = mock_session
        self._mock_query = mock_query

    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: list[float],
        depth: int,
        span_start: int,
        span_end: int,
        parent_id: Optional[str] = None,
        summary: Optional[str] = None,
        mid_offset: Optional[int] = None,
        document_id: Optional[str] = None,
        left_child_id: Optional[str] = None,
        right_child_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Add a node to the mock store."""
        # Validate embedding dimension
        if self._expected_embedding_dim is None:
            self._expected_embedding_dim = len(embedding)
        elif len(embedding) != self._expected_embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self._expected_embedding_dim}, "
                f"got {len(embedding)}"
            )

        # Create node
        node = SimpleNamespace(
            id=node_id,
            text=text,
            depth=depth,
            span_start=span_start,
            span_end=span_end,
            parent_id=parent_id,
            summary=summary,
            mid_offset=mid_offset,
            document_id=document_id,
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            is_dirty=kwargs.get("is_dirty", False),
            is_pinned=kwargs.get("is_pinned", False),
            access_count=0,
            last_accessed=None,
            created_at=None,
        )

        # Store node and embedding
        self.nodes[node_id] = node
        self.embeddings[node_id] = embedding

        # Track document association
        if document_id:
            self.document_nodes[document_id].add(node_id)

        # Add to cache
        self._add_to_cache(node)

        # Update mock session results
        self._update_mock_results()

    def get_node(self, node_id: str) -> Optional[SimpleNamespace]:
        """Get a node by ID."""
        # Check cache first
        if node_id in self.node_cache:
            # Move to end (most recently used)
            if node_id in self.cache_order:
                self.cache_order.remove(node_id)
            self.cache_order.append(node_id)
            return self.node_cache[node_id]

        # Get from storage
        node = self.nodes.get(node_id)
        if node:
            self._add_to_cache(node)
        return node

    def update_node_access(self, node_id: str) -> None:
        """Update access time and count for a node."""
        node = self.nodes.get(node_id)
        if node:
            node.access_count = getattr(node, "access_count", 0) + 1
            node.last_accessed = "mock_timestamp"

    def get_children(
        self, parent_id: str
    ) -> tuple[Optional[SimpleNamespace], Optional[SimpleNamespace]]:
        """Get children of a node."""
        parent = self.nodes.get(parent_id)
        if not parent:
            return None, None

        # Find children by parent_id
        children = [n for n in self.nodes.values() if n.parent_id == parent_id]
        children.sort(key=lambda x: x.span_start)

        left = children[0] if len(children) > 0 else None
        right = children[1] if len(children) > 1 else None

        # Alternative: use left_child_id and right_child_id if set
        if hasattr(parent, "left_child_id") and parent.left_child_id:
            left = self.nodes.get(parent.left_child_id)
        if hasattr(parent, "right_child_id") and parent.right_child_id:
            right = self.nodes.get(parent.right_child_id)

        return left, right

    def get_child(self, node_id: str, side: str) -> Optional[SimpleNamespace]:
        """Get the left or right child of a node."""
        parent = self.nodes.get(node_id)
        if not parent:
            return None

        child_id = parent.left_child_id if side == "LEFT" else parent.right_child_id
        if not child_id:
            return None

        return self.nodes.get(child_id)

    def get_nodes(self, node_ids: list[str]) -> list[SimpleNamespace]:
        """Get multiple nodes by their IDs."""
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_leaf_nodes(
        self, document_id: Optional[str] = None
    ) -> list[SimpleNamespace]:
        """Get all leaf nodes (nodes without summary)."""
        leaves = [n for n in self.nodes.values() if n.summary is None]

        if document_id:
            leaves = [n for n in leaves if n.document_id == document_id]

        return sorted(leaves, key=lambda x: x.span_start)

    def get_root_node(
        self, document_id: Optional[str] = None
    ) -> Optional[SimpleNamespace]:
        """Get the root node (node without parent)."""
        candidates = [n for n in self.nodes.values() if n.parent_id is None]

        if document_id:
            candidates = [n for n in candidates if n.document_id == document_id]

        # Return the one with highest depth (root in RagZoom's inverted convention)
        if candidates:
            return max(candidates, key=lambda x: x.depth)
        return None

    def get_root_node_for_document(
        self, document_id: Optional[str] = None
    ) -> Optional[SimpleNamespace]:
        """Get the root node for a specific document from the mock store."""
        return self.get_root_node(document_id=document_id)

    def mark_dirty_upward(self, node_id: str) -> None:
        """Mark a node and all its ancestors as dirty."""
        current_id = node_id
        while current_id:
            self.dirty_nodes.add(current_id)
            node = self.nodes.get(current_id)
            if node:
                node.is_dirty = True
                current_id = node.parent_id
            else:
                break

    def get_dirty_nodes(self) -> list[SimpleNamespace]:
        """Get all nodes marked as dirty."""
        return [self.nodes[nid] for nid in self.dirty_nodes if nid in self.nodes]

    def get_ancestors(self, node_ids: list[str]) -> list[SimpleNamespace]:
        """Get all ancestors of given nodes."""
        ancestor_ids = set()
        to_process = list(node_ids)

        while to_process:
            node_id = to_process.pop()
            node = self.nodes.get(node_id)
            if node and node.parent_id and node.parent_id not in ancestor_ids:
                ancestor_ids.add(node.parent_id)
                to_process.append(node.parent_id)

        # Return node objects, not just IDs
        return [self.nodes[aid] for aid in ancestor_ids if aid in self.nodes]

    def get_all_nodes_for_document(
        self, document_id: Optional[str]
    ) -> list[SimpleNamespace]:
        """Get all nodes for a document from the mock store."""
        if not document_id:
            return list(self.nodes.values())
        return [node for node in self.nodes.values() if node.document_id == document_id]

    def search_similar(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
        **kwargs,
    ) -> list[tuple[str, float, dict]]:
        """Simple similarity search implementation."""
        # Filter nodes based on where conditions
        eligible_nodes = list(self.nodes.keys())

        if where and "document_id" in where:
            doc_id_filter = where["document_id"]
            if isinstance(doc_id_filter, dict) and "$eq" in doc_id_filter:
                doc_id = doc_id_filter["$eq"]
                eligible_nodes = [
                    nid
                    for nid in eligible_nodes
                    if self.nodes[nid].document_id == doc_id
                ]
            elif isinstance(
                doc_id_filter, str
            ):  # Handle simple string filter for robustness
                eligible_nodes = [
                    nid
                    for nid in eligible_nodes
                    if self.nodes[nid].document_id == doc_id_filter
                ]

        results = []
        # If mock_scores is set, use it to determine the candidates and their scores
        if self.mock_scores:
            for node_id, score in self.mock_scores.items():
                if node_id in eligible_nodes:
                    metadata = {
                        "depth": self.nodes[node_id].depth,
                        "span_start": self.nodes[node_id].span_start,
                        "span_end": self.nodes[node_id].span_end,
                    }
                    results.append(
                        (node_id, 1.0 - score, metadata)
                    )  # Convert score to distance
        else:
            # Fallback to old mock behavior if no scores are set
            for i, node_id in enumerate(eligible_nodes[:n_results]):
                score = 0.9 - (i * 0.05)
                metadata = {
                    "depth": self.nodes[node_id].depth,
                    "span_start": self.nodes[node_id].span_start,
                    "span_end": self.nodes[node_id].span_end,
                }
                results.append((node_id, 1.0 - score, metadata))

        return results

    def compute_mmr_diverse_results(
        self,
        query_embedding: list[float],
        candidates: list[tuple[str, float, dict]],
        lambda_param: float,
        k: int,
    ) -> list[str]:
        """Simplified MMR implementation for testing."""
        # For testing, just return the top k candidate IDs
        # Real MMR would compute similarity between candidates
        result_ids = []
        for candidate in candidates[:k]:
            result_ids.append(candidate[0])
        return result_ids

    def pin_node(self, node_id: str) -> None:
        """Pin a node."""
        self.pinned_nodes.add(node_id)
        node = self.nodes.get(node_id)
        if node:
            node.is_pinned = True

    def set_mock_scores(self, scores: dict[str, float]):
        """Set mock scores for testing."""
        self.mock_scores = scores

    def get_pinned_nodes(
        self, max_depth: Optional[int] = None
    ) -> list[SimpleNamespace]:
        """Get all pinned nodes."""
        pinned = [self.nodes[nid] for nid in self.pinned_nodes if nid in self.nodes]

        if max_depth is not None:
            pinned = [n for n in pinned if n.depth <= max_depth]

        return pinned

    def update_summary(
        self,
        node_id: str,
        text: str,
        embedding: list[float],
        mid_offset: Optional[int] = None,
    ) -> None:
        """Update a node's summary."""
        node = self.nodes.get(node_id)
        if node:
            node.summary = text
            node.text = text
            node.mid_offset = mid_offset
            node.is_dirty = False
            self.embeddings[node_id] = embedding
            self.dirty_nodes.discard(node_id)

    def add_document(
        self,
        document_id: str,
        file_path: Optional[str],
        content_hash: str,
        chunk_count: int,
    ) -> None:
        """Add a document record."""
        self.documents[document_id] = SimpleNamespace(
            id=document_id,
            file_path=file_path,
            content_hash=content_hash,
            chunk_count=chunk_count,
        )

    def delete_document_nodes(self, document_id: str) -> None:
        """Delete all nodes for a document."""
        node_ids = list(self.document_nodes.get(document_id, []))

        for node_id in node_ids:
            self.nodes.pop(node_id, None)
            self.embeddings.pop(node_id, None)
            self.node_cache.pop(node_id, None)
            self.dirty_nodes.discard(node_id)
            self.pinned_nodes.discard(node_id)

        self.document_nodes.pop(document_id, None)
        self.documents.pop(document_id, None)

        self._update_mock_results()

    def compute_content_hash(self, content: str) -> str:
        """Compute hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()

    def find_existing_document(self, content_hash: str) -> Optional[str]:
        """Find document by content hash."""
        for doc in self.documents.values():
            if doc.content_hash == content_hash:
                return doc.id
        return None

    def close(self) -> None:
        """Close the store (no-op for mock)."""
        pass

    def _add_to_cache(self, node) -> None:
        """Add node to cache."""
        if node.id in self.node_cache:
            if node.id in self.cache_order:
                self.cache_order.remove(node.id)
        elif len(self.cache_order) >= self.cache_order.maxlen:
            # Evict LRU
            lru_id = self.cache_order.popleft()
            self.node_cache.pop(lru_id, None)

        self.node_cache[node.id] = node
        self.cache_order.append(node.id)

    def _update_mock_results(self):
        """Update mock query results based on current state."""
        all_nodes = list(self.nodes.values())
        [n for n in all_nodes if n.summary is None]

        # Setup different filter patterns
        def mock_filter_by(**kwargs):
            filtered = all_nodes
            if "document_id" in kwargs:
                filtered = [
                    n for n in filtered if n.document_id == kwargs["document_id"]
                ]
            if "is_dirty" in kwargs:
                filtered = [n for n in filtered if n.is_dirty == kwargs["is_dirty"]]
            if "parent_id" in kwargs and kwargs["parent_id"] is None:
                filtered = [n for n in filtered if n.parent_id is None]

            mock_result = MagicMock()
            mock_result.all.return_value = filtered
            mock_result.first.return_value = filtered[0] if filtered else None
            mock_result.count.return_value = len(filtered)
            mock_result.filter_by = mock_filter_by  # Make it chainable
            return mock_result

        self._mock_query.filter_by = mock_filter_by
        self._mock_query.all.return_value = all_nodes
        self._mock_query.count.return_value = len(all_nodes)
