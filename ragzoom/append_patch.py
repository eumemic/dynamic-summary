"""Append patch builder for incremental document updates."""

import uuid
from collections import deque
from dataclasses import dataclass, field

from ragzoom.contracts.tree_node import TreeNode, get_depth
from ragzoom.dataflow.core import TreePatch
from ragzoom.dataflow.domain import DomainNode
from ragzoom.document_store import DocumentStore
from ragzoom.utils.tokenization import TokenizerUtil


@dataclass
class PatchTracking:
    """Tracking information for incremental append patches."""

    mutable_node_ids: set[str]
    context_node_ids: set[str]
    original_neighbors: dict[str, tuple[str | None, str | None]]
    neighbor_updates: list[tuple[str, str | None, str | None]] = field(
        default_factory=list
    )
    leaf_delta: int = 0
    tail_start: int = 0
    tail_text: str = ""
    summary_node_ids: set[str] = field(default_factory=set)
    original_heights: dict[str, int] = field(default_factory=dict)


class AppendPatchBuilder:
    """Builds TreePatch objects for incremental document appends.

    This class encapsulates the logic for constructing patches when appending
    new text to an existing document. It handles spine traversal, leaf creation,
    neighbor linking, and parent level building.
    """

    def __init__(
        self,
        document_store: DocumentStore,
        tokenizer: TokenizerUtil,
    ) -> None:
        """Initialize the patch builder.

        Args:
            document_store: Storage for node lookups during spine collection.
            tokenizer: Tokenizer for counting tokens in new chunks.
        """
        self.document_store = document_store
        self.tokenizer = tokenizer

    def build_patch(
        self,
        right_leaf: TreeNode,
        new_chunks: list[str],
        document_id: str,
    ) -> tuple[TreePatch, PatchTracking]:
        """Build a TreePatch for incremental append.

        Args:
            right_leaf: The rightmost leaf node (anchor point for append).
            new_chunks: Text chunks to append (already split/validated).
            document_id: Target document identifier.

        Returns:
            Tuple of (TreePatch, PatchTracking) containing the patch structure
            and tracking metadata for persistence and rollback.

        Raises:
            ValueError: If inputs are invalid.
        """
        lookup: dict[str, DomainNode] = {}
        tracking = PatchTracking(set(), set(), {})

        spine_nodes = self._collect_spine(right_leaf)
        self._validate_append_inputs(new_chunks, spine_nodes)

        spine_domains = self._collect_spine_domains(spine_nodes, lookup, tracking)

        leaf_domain = spine_domains[right_leaf.id]
        tracking.tail_start = int(leaf_domain.span_start)
        original_following = leaf_domain.following_neighbor_id
        self._ensure_context_nodes(
            lookup, [leaf_domain.preceding_neighbor_id], tracking
        )

        new_leaf_domains = self._create_leaf_domains(
            leaf_domain, new_chunks, document_id, lookup, tracking
        )
        tracking.tail_text = "".join(new_chunks)

        last_leaf_id = self._link_leaf_neighbors(
            leaf_domain, new_leaf_domains, original_following
        )
        tracking.leaf_delta = len(new_chunks) - 1

        self._handle_following_neighbor(original_following, last_leaf_id, tracking)

        embedding_ids = [leaf_domain.id] + [leaf.id for leaf in new_leaf_domains]

        current_level = self._initialize_current_level(
            spine_nodes, right_leaf, leaf_domain, new_leaf_domains, lookup, tracking
        )

        summary_root_ids: list[str] = []

        # Traverse up the spine, reusing existing parents when present
        for level_index in range(len(spine_nodes) - 1):
            parent_tree = spine_nodes[level_index + 1]
            parent_domain = spine_domains[parent_tree.id]

            next_level, summary_ids = self._build_parent_level(
                current_level,
                parent_domain,
                document_id,
                lookup,
                tracking,
            )
            summary_root_ids.extend(summary_ids)
            current_level = next_level

            self._inject_left_sibling_if_needed(
                spine_nodes, level_index, parent_domain, current_level, lookup, tracking
            )

        current_level = self._build_additional_parents(
            current_level, document_id, lookup, tracking, summary_root_ids
        )

        if current_level:
            current_level[0].parent_id = None
            self._assign_patch_depths(current_level[0].id, lookup)

        patch = TreePatch(
            lookup=lookup,
            embedding_node_ids=embedding_ids,
            summary_root_ids=summary_root_ids,
        )
        return patch, tracking

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return str(uuid.uuid4())

    def _node_to_domain(self, node: TreeNode) -> DomainNode:
        """Convert a stored TreeNode into a DomainNode for patch construction."""
        if node.document_id is None:
            if self.document_store.document_id is None:
                raise ValueError(
                    f"Node {node.id} has no document_id and DocumentStore has no default"
                )
            document_id = self.document_store.document_id
        else:
            document_id = node.document_id

        is_pinned_raw = getattr(node, "is_pinned", False)
        if isinstance(is_pinned_raw, int):
            is_pinned = bool(is_pinned_raw)
        else:
            is_pinned = bool(is_pinned_raw)

        return DomainNode(
            id=node.id,
            document_id=document_id,
            parent_id=node.parent_id,
            left_child_id=node.left_child_id,
            right_child_id=node.right_child_id,
            span_start=int(node.span_start),
            span_end=int(node.span_end),
            text=node.text,
            token_count=int(node.token_count),
            height=int(node.height),
            is_pinned=is_pinned,
            depth=get_depth(node),
            preceding_neighbor_id=node.preceding_neighbor_id,
            following_neighbor_id=node.following_neighbor_id,
            embedding=None,
            level_index=int(node.level_index),
        )

    def _collect_spine(self, leaf: TreeNode) -> list[TreeNode]:
        """Collect nodes along the rightmost spine starting from the provided leaf."""
        spine: list[TreeNode] = [leaf]
        current = leaf
        while current.parent_id:
            parent = self.document_store.nodes.get(current.parent_id)
            if parent is None:
                raise ValueError(
                    "Encountered missing ancestor while tracing right spine"
                )
            spine.append(parent)
            current = parent
        return spine

    def _ensure_context_nodes(
        self,
        lookup: dict[str, DomainNode],
        candidates: list[str | None],
        tracking: PatchTracking,
    ) -> None:
        """Ensure referenced neighbor nodes are present in the patch lookup."""
        for node_id in candidates:
            if not node_id:
                continue
            if node_id in lookup:
                tracking.context_node_ids.add(node_id)
                continue
            node = self.document_store.nodes.get(node_id)
            if node is None:
                continue
            domain = self._node_to_domain(node)
            lookup[domain.id] = domain
            tracking.context_node_ids.add(domain.id)
            tracking.original_neighbors.setdefault(
                domain.id,
                (domain.preceding_neighbor_id, domain.following_neighbor_id),
            )

    def _load_sibling_domain(
        self,
        sibling_id: str,
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
    ) -> DomainNode | None:
        """Load a sibling node into the lookup and register for context tracking.

        Returns the sibling DomainNode if found, None otherwise.
        """
        sibling_domain = lookup.get(sibling_id)
        if sibling_domain is None:
            sibling_node = self.document_store.nodes.get(sibling_id)
            if sibling_node is not None:
                sibling_domain = self._node_to_domain(sibling_node)
                lookup[sibling_domain.id] = sibling_domain
            if sibling_domain is not None:
                tracking.context_node_ids.add(sibling_domain.id)
                tracking.original_neighbors.setdefault(
                    sibling_domain.id,
                    (
                        sibling_domain.preceding_neighbor_id,
                        sibling_domain.following_neighbor_id,
                    ),
                )
        return sibling_domain

    def _validate_append_inputs(
        self,
        new_chunks: list[str],
        spine_nodes: list[TreeNode],
    ) -> None:
        """Validate inputs for append patch construction.

        Raises:
            ValueError: If new_chunks is empty or spine is missing.
        """
        if not new_chunks:
            raise ValueError("Append requires at least one chunk of text")
        if not spine_nodes:
            raise ValueError("Rightmost leaf is missing its ancestor chain")

    def _collect_spine_domains(
        self,
        spine_nodes: list[TreeNode],
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
    ) -> dict[str, DomainNode]:
        """Convert spine TreeNodes to DomainNodes and register for tracking.

        Returns:
            Dictionary mapping node IDs to spine DomainNodes.
        """
        spine_domains: dict[str, DomainNode] = {}
        for node in spine_nodes:
            domain = self._node_to_domain(node)
            lookup[domain.id] = domain
            spine_domains[domain.id] = domain
            tracking.mutable_node_ids.add(domain.id)
            tracking.original_neighbors[domain.id] = (
                domain.preceding_neighbor_id,
                domain.following_neighbor_id,
            )
            tracking.original_heights[domain.id] = int(domain.height)
        return spine_domains

    def _create_leaf_domains(
        self,
        leaf_domain: DomainNode,
        new_chunks: list[str],
        document_id: str,
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
    ) -> list[DomainNode]:
        """Create new leaf DomainNodes from chunks after the first.

        Updates leaf_domain in place with first chunk.
        Creates new DomainNode for each subsequent chunk.

        Returns:
            List of newly created leaf DomainNodes (excludes updated original).
        """
        new_leaf_domains: list[DomainNode] = []
        span_cursor = leaf_domain.span_start
        for idx, chunk in enumerate(new_chunks):
            span_end = span_cursor + len(chunk)
            token_count = self.tokenizer.count_tokens(chunk)

            if idx == 0:
                leaf_domain.text = chunk
                leaf_domain.span_start = span_cursor
                leaf_domain.span_end = span_end
                leaf_domain.token_count = token_count
            else:
                new_leaf = DomainNode(
                    id=self._generate_node_id(),
                    document_id=document_id,
                    parent_id=None,
                    left_child_id=None,
                    right_child_id=None,
                    span_start=span_cursor,
                    span_end=span_end,
                    text=chunk,
                    token_count=token_count,
                    height=0,
                    preceding_neighbor_id=None,
                    following_neighbor_id=None,
                )
                lookup[new_leaf.id] = new_leaf
                new_leaf_domains.append(new_leaf)
                tracking.mutable_node_ids.add(new_leaf.id)
                tracking.original_neighbors[new_leaf.id] = (None, None)

            span_cursor = span_end

        return new_leaf_domains

    def _link_leaf_neighbors(
        self,
        leaf_domain: DomainNode,
        new_leaf_domains: list[DomainNode],
        original_following: str | None,
    ) -> str:
        """Establish neighbor links between leaf nodes.

        Returns:
            ID of the last leaf in the chain.
        """
        last_leaf_id = leaf_domain.id
        if new_leaf_domains:
            leaf_domain.following_neighbor_id = new_leaf_domains[0].id
            for idx, leaf in enumerate(new_leaf_domains):
                leaf.preceding_neighbor_id = (
                    new_leaf_domains[idx - 1].id if idx > 0 else leaf_domain.id
                )
                leaf.following_neighbor_id = (
                    new_leaf_domains[idx + 1].id
                    if idx + 1 < len(new_leaf_domains)
                    else original_following
                )
            last_leaf_id = new_leaf_domains[-1].id
        else:
            leaf_domain.following_neighbor_id = original_following
        return last_leaf_id

    def _handle_following_neighbor(
        self,
        original_following: str | None,
        last_leaf_id: str,
        tracking: PatchTracking,
    ) -> None:
        """Record neighbor updates for rollback when following neighbor exists."""
        if not original_following:
            return

        following_node = self.document_store.nodes.get(original_following)
        following_follow = (
            getattr(following_node, "following_neighbor_id", None)
            if following_node
            else None
        )
        # Record how the right-edge neighbor chain is rewritten so rollback can restore it
        tracking.neighbor_updates.append(
            (original_following, last_leaf_id, following_follow)
        )
        if following_node is not None:
            tracking.context_node_ids.add(original_following)
            tracking.original_neighbors.setdefault(
                original_following,
                (
                    getattr(following_node, "preceding_neighbor_id", None),
                    getattr(following_node, "following_neighbor_id", None),
                ),
            )

    def _initialize_current_level(
        self,
        spine_nodes: list[TreeNode],
        right_leaf: TreeNode,
        leaf_domain: DomainNode,
        new_leaf_domains: list[DomainNode],
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
    ) -> list[DomainNode]:
        """Initialize current level with left sibling if right child, plus leaves."""
        current_level: list[DomainNode] = []
        # Include left sibling when the path node is a right child
        if len(spine_nodes) > 1:
            parent_tree = spine_nodes[1]
            if parent_tree.right_child_id == right_leaf.id:
                left_sibling_id = parent_tree.left_child_id
                if left_sibling_id:
                    sibling_domain = self._load_sibling_domain(
                        left_sibling_id, lookup, tracking
                    )
                    if sibling_domain is not None:
                        current_level.append(sibling_domain)
                        self._ensure_context_nodes(
                            lookup,
                            [sibling_domain.preceding_neighbor_id],
                            tracking,
                        )

        current_level.append(leaf_domain)
        current_level.extend(new_leaf_domains)
        return current_level

    def _inject_left_sibling_if_needed(
        self,
        spine_nodes: list[TreeNode],
        level_index: int,
        parent_domain: DomainNode,
        current_level: list[DomainNode],
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
    ) -> None:
        """Inject left sibling into current level during spine traversal."""
        if level_index + 1 >= len(spine_nodes) - 1:
            return

        next_parent_tree = spine_nodes[level_index + 2]
        if next_parent_tree.right_child_id != parent_domain.id:
            return

        left_sibling_id = next_parent_tree.left_child_id
        if not left_sibling_id:
            return

        sibling_domain = self._load_sibling_domain(left_sibling_id, lookup, tracking)

        if sibling_domain is not None and all(
            sibling_domain.id != existing.id for existing in current_level
        ):
            # Pull the left sibling into the patch to keep adjacency consistent
            current_level.insert(0, sibling_domain)
            self._ensure_context_nodes(
                lookup,
                [sibling_domain.preceding_neighbor_id],
                tracking,
            )

    def _build_additional_parents(
        self,
        current_level: list[DomainNode],
        document_id: str,
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
        summary_root_ids: list[str],
    ) -> list[DomainNode]:
        """Build additional parent levels if new nodes extended tree height."""
        while len(current_level) > 1:
            current_level, summary_ids = self._build_parent_level(
                current_level,
                None,
                document_id,
                lookup,
                tracking,
            )
            summary_root_ids.extend(summary_ids)
        return current_level

    def _build_parent_level(
        self,
        nodes: list[DomainNode],
        existing_parent: DomainNode | None,
        document_id: str,
        lookup: dict[str, DomainNode],
        tracking: PatchTracking,
    ) -> tuple[list[DomainNode], list[str]]:
        """Pair nodes into parents, reusing spine ancestor when provided."""
        if not nodes:
            return [], []

        summary_ids: list[str] = []
        next_level: list[DomainNode] = []
        idx = 0
        reuse_available = existing_parent is not None

        while idx < len(nodes):
            left = nodes[idx]
            right = nodes[idx + 1] if idx + 1 < len(nodes) else None

            if reuse_available:
                assert existing_parent is not None
                parent = existing_parent
                reuse_available = False
            else:
                parent = DomainNode(
                    id=self._generate_node_id(),
                    document_id=document_id,
                    parent_id=None,
                    left_child_id=None,
                    right_child_id=None,
                    span_start=0,
                    span_end=0,
                    text="",
                    token_count=0,
                    height=0,
                )
                lookup[parent.id] = parent
                tracking.mutable_node_ids.add(parent.id)
                tracking.original_neighbors[parent.id] = (
                    parent.preceding_neighbor_id,
                    parent.following_neighbor_id,
                )

            parent.document_id = document_id
            parent.left_child_id = left.id
            parent.span_start = left.span_start
            left.parent_id = parent.id

            if right is not None:
                parent.right_child_id = right.id
                parent.span_end = right.span_end
                parent.height = max(int(left.height), int(right.height)) + 1
                right.parent_id = parent.id
                step = 2
            else:
                parent.right_child_id = None
                parent.span_end = left.span_end
                parent.height = int(left.height) + 1
                step = 1

            parent.text = ""
            parent.token_count = 0
            parent.embedding = None

            summary_ids.append(parent.id)
            tracking.summary_node_ids.add(parent.id)
            self._ensure_context_nodes(lookup, [left.preceding_neighbor_id], tracking)

            next_level.append(parent)

            idx += step

        # Update neighbor links for the newly formed level
        for i, parent in enumerate(next_level):
            if i > 0:
                parent.preceding_neighbor_id = next_level[i - 1].id
            if i + 1 < len(next_level):
                parent.following_neighbor_id = next_level[i + 1].id

        return next_level, summary_ids

    # jscpd:ignore-start
    # Similar to _assign_depths in dataflow/core.py but operates on DomainNode
    # instead of TreeNode. Both types share the same structural fields but are
    # distinct types used in different contexts (patch building vs tree execution).
    def _assign_patch_depths(self, root_id: str, lookup: dict[str, DomainNode]) -> None:
        """Assign depth values within a patch so batching logic remains correct."""
        queue: deque[tuple[str, int]] = deque([(root_id, 0)])
        visited: set[str] = set()

        while queue:
            node_id, depth = queue.popleft()
            node = lookup.get(node_id)
            if node is None or node_id in visited:
                continue
            visited.add(node_id)
            node.depth = depth

            if node.left_child_id:
                queue.append((node.left_child_id, depth + 1))
            if node.right_child_id:
                queue.append((node.right_child_id, depth + 1))

    # jscpd:ignore-end
