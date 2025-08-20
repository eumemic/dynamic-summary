"""Mock Store implementation for fast testing."""

import hashlib
from collections import OrderedDict, defaultdict, deque
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock


class SimpleMockStore:
    """Lightweight mock of Store for unit testing.

    This mock provides fast, in-memory implementations of the most commonly
    used Store methods. It maintains consistency for tree operations and
    basic state management without the overhead of real database I/O.
    """

    PIN_DEPTH_MAX = 2  # Match Store class constant

    def __init__(self, config=None):
        self.config = config
        self.nodes: dict[str, SimpleNamespace] = {}
        self.embeddings: dict[str, list[float]] = {}
        self.document_nodes: dict[str, set[str]] = defaultdict(set)
        self.documents: dict[str, SimpleNamespace] = {}

        # State tracking
        self.pinned_nodes: set[str] = set()
        self.mock_scores: dict[str, float] = {}

        # Cache simulation
        self._node_cache = OrderedDict()
        self._cache_order = deque(maxlen=1000)

        # Mock SessionLocal for tests that use direct DB access
        self._setup_session_mock()

        # Track expected embedding dimension
        self._expected_embedding_dim = None

    def _setup_session_mock(self):
        """Setup mock session for compatibility with tests using SessionLocal."""
        mock_session = MagicMock()

        # Create a proper query mock that can handle filter_by and update nodes
        def create_query_mock(model_class):
            # Determine the model type
            model_name = getattr(model_class, "__name__", str(model_class))

            query_mock = MagicMock()

            # Set up model-specific all() and count() methods based on the model type
            if "TreeNode" in model_name or "Node" in model_name:
                query_mock.all = MagicMock(return_value=list(self.nodes.values()))
                query_mock.count = MagicMock(return_value=len(self.nodes))
                query_mock.first = MagicMock(
                    return_value=list(self.nodes.values())[0] if self.nodes else None
                )
            elif "Document" in model_name:
                query_mock.all = MagicMock(return_value=list(self.documents.values()))
                query_mock.count = MagicMock(return_value=len(self.documents))
                query_mock.first = MagicMock(
                    return_value=(
                        list(self.documents.values())[0] if self.documents else None
                    )
                )
            else:
                # Default fallback
                query_mock.all = MagicMock(return_value=[])
                query_mock.count = MagicMock(return_value=0)
                query_mock.first = MagicMock(return_value=None)

            def filter_by_impl(**kwargs):
                # Handle TreeNode queries - check for TreeNode or similar patterns
                if "TreeNode" in model_name or "Node" in model_name:
                    if "id" in kwargs:
                        node_id = kwargs["id"]
                        # Return a mock that will find our node
                        result_mock = MagicMock()

                        def first_impl():
                            # Return the actual node from our store
                            if node_id in self.nodes:
                                return self.nodes[node_id]
                            return None

                        result_mock.first = first_impl
                        return result_mock
                    elif "document_id" in kwargs:
                        doc_id = kwargs["document_id"]
                        parent_id = kwargs.get("parent_id")
                        result_mock = MagicMock()

                        def all_impl():
                            nodes = [
                                node
                                for node in self.nodes.values()
                                if node.document_id == doc_id
                            ]
                            if parent_id is not None:
                                nodes = [
                                    node
                                    for node in nodes
                                    if node.parent_id == parent_id
                                ]
                            return nodes

                        def count_impl():
                            nodes = all_impl()
                            return len(nodes)

                        def first_impl():
                            nodes = all_impl()
                            return nodes[0] if nodes else None

                        result_mock.all = all_impl
                        result_mock.count = count_impl
                        result_mock.first = first_impl
                        return result_mock
                    else:
                        result_mock = MagicMock()
                        result_mock.all.return_value = list(self.nodes.values())
                        result_mock.count.return_value = len(self.nodes)
                        return result_mock

                # Handle Document queries
                elif "Document" in model_name:
                    if "id" in kwargs:
                        doc_id = kwargs["id"]
                        result_mock = MagicMock()

                        def first_impl():
                            return self.documents.get(doc_id)

                        result_mock.first = first_impl
                        return result_mock
                    else:
                        # Handle .all() query for listing all documents
                        result_mock = MagicMock()
                        result_mock.all.return_value = list(self.documents.values())
                        result_mock.count.return_value = len(self.documents)
                        result_mock.first.return_value = (
                            list(self.documents.values())[0] if self.documents else None
                        )
                        return result_mock

                # Default fallback
                result_mock = MagicMock()
                result_mock.all.return_value = []
                result_mock.count.return_value = 0
                result_mock.first.return_value = None
                return result_mock

            query_mock.filter_by = filter_by_impl
            query_mock.filter = MagicMock(return_value=query_mock)

            return query_mock

        mock_session.query = create_query_mock
        mock_session.commit = MagicMock()  # No-op for commits
        mock_session.__enter__ = MagicMock(return_value=mock_session)
        mock_session.__exit__ = MagicMock(return_value=None)

        self.SessionLocal = MagicMock(return_value=mock_session)

    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: list[float],
        span_start: int,
        span_end: int,
        parent_id: str | None = None,
        document_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
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

        # Create node - compute token_count if not provided
        import tiktoken

        if "token_count" not in kwargs:
            tokenizer = tiktoken.get_encoding("cl100k_base")
            token_count = len(tokenizer.encode(text))
        else:
            token_count = kwargs.get("token_count")

        node = SimpleNamespace(
            id=node_id,
            text=text,
            span_start=span_start,
            span_end=span_end,
            parent_id=parent_id,
            document_id=document_id,
            left_child_id=left_child_id,
            right_child_id=right_child_id,
            token_count=token_count,
            preceding_neighbor_id=kwargs.get("preceding_neighbor_id"),
            is_pinned=kwargs.get("is_pinned", False),
            access_count=0,
            last_accessed=None,
            created_at=None,
            embedding=embedding,  # Include embedding in the node
            height=kwargs.get("height", 0),  # Add height support
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

        return node

    def add_nodes_batch(
        self, nodes_data: list[dict], *, session=None
    ) -> list[SimpleNamespace]:
        """Add multiple nodes in batch - mock implementation."""
        created_nodes = []
        for data in nodes_data:
            # Create node using add_node for consistency
            self.add_node(
                node_id=data["node_id"],
                text=data["text"],
                embedding=data["embedding"],
                span_start=data["span_start"],
                span_end=data["span_end"],
                parent_id=data.get("parent_id"),
                left_child_id=data.get("left_child_id"),
                right_child_id=data.get("right_child_id"),
                document_id=data.get("document_id"),
                token_count=data.get("token_count"),
                preceding_neighbor_id=data.get("preceding_neighbor_id"),
                height=data.get("height", 0),  # Pass through height parameter
            )
            created_nodes.append(self.nodes[data["node_id"]])
        return created_nodes

    def update_parent_references_batch(
        self, updates: list[tuple[str, str]], *, session=None
    ) -> None:
        """Update parent references in batch - mock implementation."""
        for child_id, parent_id in updates:
            if child_id in self.nodes:
                self.nodes[child_id].parent_id = parent_id
                # Invalidate cache for updated node
                if child_id in self._node_cache:
                    del self._node_cache[child_id]
                    if child_id in self._cache_order:
                        self._cache_order.remove(child_id)

    def get_node(self, node_id: str) -> SimpleNamespace | None:
        """Get a node by ID."""
        # Check cache first
        if node_id in self._node_cache:
            # Move to end (most recently used)
            if node_id in self._cache_order:
                self._cache_order.remove(node_id)
            self._cache_order.append(node_id)
            return self._node_cache[node_id]

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
    ) -> tuple[SimpleNamespace | None, SimpleNamespace | None]:
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

    def get_nodes(self, node_ids: list[str]) -> list[SimpleNamespace]:
        """Get multiple nodes by their IDs."""
        return [self.nodes[nid] for nid in node_ids if nid in self.nodes]

    def get_leaf_nodes(self, document_id: str | None = None) -> list[SimpleNamespace]:
        """Get all leaf nodes (nodes without children)."""
        leaves = [
            n
            for n in self.nodes.values()
            if n.left_child_id is None and n.right_child_id is None
        ]

        if document_id:
            leaves = [n for n in leaves if n.document_id == document_id]

        return sorted(leaves, key=lambda x: x.span_start)

    def get_root_node(self, document_id: str | None = None) -> SimpleNamespace | None:
        """Get the root node (node without parent)."""
        candidates = [n for n in self.nodes.values() if n.parent_id is None]

        if document_id:
            candidates = [n for n in candidates if n.document_id == document_id]

        # Return any root node (there should only be one per document)
        if candidates:
            return candidates[0]
        return None

    def get_root_node_for_document(
        self, document_id: str | None = None
    ) -> SimpleNamespace | None:
        """Get the root node for a specific document from the mock store."""
        return self.get_root_node(document_id=document_id)

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

    def get_node_depth(self, node_id: str) -> int:
        """Calculate depth of a node (distance from root)."""
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        depth = 0
        current_id = node.parent_id

        while current_id:
            depth += 1
            parent = self.get_node(current_id)
            if not parent:
                break
            current_id = parent.parent_id

        return depth

    def is_leaf_node(self, node_id: str) -> bool:
        """Check if a node is a leaf (has no children)."""
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        return not node.left_child_id and not node.right_child_id

    def is_root_node(self, node_id: str) -> bool:
        """Check if a node is a root (has no parent)."""
        node = self.get_node(node_id)
        if not node:
            raise ValueError(f"Node {node_id} not found")

        return node.parent_id is None

    def get_all_nodes(self) -> list[SimpleNamespace]:
        """Get all nodes from the mock store."""
        return list(self.nodes.values())

    def get_all_nodes_for_document(
        self, document_id: str | None
    ) -> list[SimpleNamespace]:
        """Get all nodes for a document from the mock store."""
        if not document_id:
            return list(self.nodes.values())
        return [node for node in self.nodes.values() if node.document_id == document_id]

    def search_similar(
        self,
        query_embedding: list[float],
        n_results: int = 10,
        where: dict | None = None,
        where_document: dict | None = None,
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
                        "depth": self.get_node_depth(node_id),
                        "span_start": self.nodes[node_id].span_start,
                        "span_end": self.nodes[node_id].span_end,
                    }
                    results.append(
                        (node_id, 1.0 - score, metadata)
                    )  # Convert score to distance
        else:
            # Calculate similarity based on embeddings
            for node_id in eligible_nodes:
                node_embedding = self.embeddings.get(
                    node_id, [0.5] * len(query_embedding)
                )
                # Simple similarity: compare first element
                similarity = 1.0 - abs(query_embedding[0] - node_embedding[0])
                distance = 1.0 - similarity
                metadata = {
                    "depth": self.get_node_depth(node_id),
                    "span_start": self.nodes[node_id].span_start,
                    "span_end": self.nodes[node_id].span_end,
                }
                results.append((node_id, distance, metadata))

            # Sort by distance (ascending) and take top n_results
            results.sort(key=lambda x: x[1])
            results = results[:n_results]

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

    def get_document_by_path(self, file_path: str):
        """Mock get document by path."""
        # Check documents by file_path
        for doc in self.documents.values():
            if hasattr(doc, "file_path") and doc.file_path == file_path:
                return doc
        return None

    def get_document_by_id(self, document_id: str):
        """Mock get document by ID."""
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
        """Mock add document."""
        from datetime import datetime
        from types import SimpleNamespace

        doc = SimpleNamespace(
            id=document_id,
            file_path=file_path,
            content_hash=content_hash,
            chunk_count=chunk_count,
            embedding_model=embedding_model,
            summary_model=summary_model,
            indexed_at=datetime.utcnow(),
        )
        self.documents[document_id] = doc
        return doc

    @staticmethod
    def compute_content_hash(content: str) -> str:
        """Mock content hash computation."""

        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def pin_node(self, node_id: str) -> None:
        """Pin a node."""
        self.pinned_nodes.add(node_id)
        node = self.nodes.get(node_id)
        if node:
            node.is_pinned = True

    def set_mock_scores(self, scores: dict[str, float]):
        """Set mock scores for testing."""
        self.mock_scores = scores

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[SimpleNamespace]:
        """Get all pinned nodes."""
        pinned = [self.nodes[nid] for nid in self.pinned_nodes if nid in self.nodes]

        if depth_max is not None:
            pinned = [n for n in pinned if self.get_node_depth(n.id) <= depth_max]

        return pinned

    def update_text(
        self,
        node_id: str,
        text: str,
        embedding: list[float],
    ) -> None:
        """Update a node's text content."""
        node = self.nodes.get(node_id)
        if node:
            node.text = text
            self.embeddings[node_id] = embedding

    def get_document_embedding_model(self, document_id: str) -> str | None:
        """Get the embedding model used for a specific document."""
        doc = self.get_document_by_id(document_id)
        return doc.embedding_model if doc and hasattr(doc, "embedding_model") else None

    def delete_document_nodes(self, document_id: str, *, session=None) -> int:
        """Delete all nodes for a document."""
        node_ids = list(self.document_nodes.get(document_id, []))
        deleted_count = len(node_ids)

        for node_id in node_ids:
            self.nodes.pop(node_id, None)
            self.embeddings.pop(node_id, None)
            self._node_cache.pop(node_id, None)
            self.pinned_nodes.discard(node_id)

        self.document_nodes.pop(document_id, None)
        self.documents.pop(document_id, None)

        self._update_mock_results()
        return deleted_count

    def find_existing_document(self, content_hash: str) -> str | None:
        """Find document by content hash."""
        for doc in self.documents.values():
            if doc.content_hash == content_hash:
                return doc.id
        return None

    def get_document_token_stats(self, document_id: str) -> dict[str, float | int]:
        """Get token statistics for a document."""
        return {
            "avg_tokens": 100.0,
            "min_tokens": 50,
            "max_tokens": 150,
            "total_tokens": 1000,
            "node_count": 10,
        }

    def clear_document(self, document_id: str, *, session=None) -> int:
        """Clear all data for a document."""
        # Count nodes before deleting
        node_count = len(
            [n for n in self.nodes.values() if n.document_id == document_id]
        )
        self.delete_document_nodes(document_id, session=session)
        return node_count

    @contextmanager
    def transaction(self):
        """Mock transaction context manager.

        For the mock store, this doesn't provide true transactional behavior
        but allows testing transaction-aware code paths.
        """
        # Create a mock session object for compatibility
        mock_session = MagicMock()
        yield mock_session

    def close(self) -> None:
        """Close the store (no-op for mock)."""
        pass

    # Add cache attributes for CLI compatibility
    @property
    def node_cache(self):
        """Mock node_cache for CLI compatibility."""
        return self._node_cache if hasattr(self, "_node_cache") else {}

    @property
    def cache_order(self):
        """Mock cache_order for CLI compatibility."""
        return self._cache_order if hasattr(self, "_cache_order") else []

    def _add_to_cache(self, node) -> None:
        """Add node to cache."""
        if node.id in self._node_cache:
            if node.id in self._cache_order:
                self._cache_order.remove(node.id)
        elif len(self._cache_order) >= self._cache_order.maxlen:
            # Evict LRU
            lru_id = self._cache_order.popleft()
            self._node_cache.pop(lru_id, None)

        self._node_cache[node.id] = node
        self._cache_order.append(node.id)

    def _update_mock_results(self):
        """Update mock query results based on current state."""
        all_nodes = list(self.nodes.values())

        # Setup different filter patterns
        def mock_filter_by(**kwargs):
            filtered = all_nodes
            if "document_id" in kwargs:
                filtered = [
                    n for n in filtered if n.document_id == kwargs["document_id"]
                ]
            if "parent_id" in kwargs and kwargs["parent_id"] is None:
                filtered = [n for n in filtered if n.parent_id is None]

            mock_result = MagicMock()
            mock_result.all.return_value = filtered
            mock_result.first.return_value = filtered[0] if filtered else None
            mock_result.count.return_value = len(filtered)
            mock_result.filter_by = mock_filter_by  # Make it chainable
            return mock_result

        # No longer needed with the new session mock implementation
        pass
