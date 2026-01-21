"""Protocol for node repository used by DocumentStore and services.

Defines the minimal, backend-agnostic surface area required by core code.
Implementations include Postgres and SQLite repositories.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Literal, Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Required

# jscpd:ignore-start - Protocol declares signatures mirrored by implementations
from ragzoom.contracts.tree_node import TreeNode

if TYPE_CHECKING:
    from ragzoom.validation.types import SQLValidationResult

# Value types that can appear in node data fields
NodeFieldValue = str | int | float | bool | list[float] | NDArray[np.float64] | None


class NodeDataDict(TypedDict, total=False):
    """Type definition for node data used in batch operations.

    Required fields are accessed directly in implementations.
    Optional fields are accessed with .get() and may be None.
    """

    # Required fields (always accessed directly in implementations)
    node_id: Required[str]
    text: Required[str]
    span_start: Required[int]
    span_end: Required[int]
    token_count: Required[int]
    height: Required[int]
    level_index: Required[int]

    # Optional fields (accessed with .get())
    document_id: str | None
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None

    # Contextual indexing fields
    preceding_context: str | None
    preceding_context_summary: str | None

    # Cost in USD for creating this node
    cost: float | None

    # Temporal metadata (Unix float seconds)
    time_start: float | None
    time_end: float | None


try:  # Optional typing import; not required at runtime
    from typing import TYPE_CHECKING

    if TYPE_CHECKING:
        from sqlalchemy.orm import Session
except Exception:  # pragma: no cover - types only
    pass


@runtime_checkable
class NodeRepository(Protocol):
    # Creation
    def add_node(
        self,
        node_id: str,
        text: str,
        embedding: list[float],
        span_start: int,
        span_end: int,
        parent_id: str | None = None,
        left_child_id: str | None = None,
        right_child_id: str | None = None,
        document_id: str | None = None,
        token_count: int = 0,
        height: int = 0,
        is_left_child: bool | None = None,
        level_index: int = 0,
    ) -> TreeNode: ...

    def add_nodes_batch(
        self,
        nodes_data: list[NodeDataDict],
        *,
        session: Session | None = None,
    ) -> list[TreeNode]: ...

    def upsert_nodes_batch(
        self,
        nodes_data: list[NodeDataDict],
        *,
        session: Session | None = None,
    ) -> list[TreeNode]: ...

    # Reads
    def get_node(self, node_id: str) -> TreeNode | None: ...
    def get_nodes(self, node_ids: list[str]) -> list[TreeNode]: ...
    def get_all_nodes_for_document(self, document_id: str | None) -> list[TreeNode]: ...
    def get_all_nodes_for_document_paginated(
        self, document_id: str | None, *, page_size: int = 1000
    ) -> list[list[TreeNode]]: ...
    def get_root_nodes(self, document_id: str | None = None) -> list[TreeNode]: ...

    # Iterators for streaming access (avoid loading all nodes into memory)
    def iter_root_nodes_for_document(
        self, document_id: str | None
    ) -> Iterator[TreeNode]:
        """Iterate over root nodes ordered by span_start.

        Uses server-side cursor to avoid loading all nodes into memory.
        """
        ...

    def iter_leaves_for_document(self, document_id: str | None) -> Iterator[TreeNode]:
        """Iterate over leaf nodes ordered by span_start.

        Uses server-side cursor to avoid loading all nodes into memory.
        """
        ...

    def iter_all_for_document(self, document_id: str | None) -> Iterator[TreeNode]:
        """Iterate over all nodes ordered by span_start.

        Uses server-side cursor to avoid loading all nodes into memory.
        """
        ...

    # Aggregations
    def count_nodes_for_document(self, document_id: str | None) -> int: ...
    def get_leaf_nodes(self) -> list[TreeNode]: ...
    def count_leaves_for_document(self, document_id: str | None) -> int: ...
    def count_leaves_with_embeddings_for_document(self, document_id: str) -> int: ...
    def get_recent_leaves_within_budget(
        self, document_id: str | None, token_budget: int
    ) -> list[TreeNode]: ...
    def get_recent_leaves_within_budget_before(
        self, document_id: str, token_budget: int, before_span_end: int
    ) -> list[TreeNode]: ...
    def max_height_for_document(self, document_id: str | None) -> int: ...
    def sum_leaf_tokens_for_document(self, document_id: str | None) -> int:
        """Return sum of token_count for all leaves in document."""
        ...

    def sum_root_tokens_for_document(self, document_id: str | None) -> int:
        """Return sum of token_count for all root nodes in document."""
        ...

    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]: ...
    def get_pinned_nodes_for_document(
        self, document_id: str, depth_max: int | None = None
    ) -> list[TreeNode]: ...
    def get_parentless_nodes_for_document(
        self, document_id: str | None
    ) -> list[TreeNode]: ...
    def get_ready_left_children(self, document_id: str | None) -> list[str]: ...
    def get_node_by_height_and_level(
        self,
        document_id: str | None,
        height: int,
        level_index: int,
    ) -> TreeNode | None: ...

    def get_nodes_by_height_levels(
        self,
        document_id: str | None,
        coordinates: Sequence[tuple[int, int]],
    ) -> list[TreeNode]: ...

    # Mutations
    def update_parent_references_batch(
        self,
        updates: Sequence[tuple[str, str | None]],
        *,
        session: Session | None = None,
    ) -> None: ...

    # jscpd:ignore-start
    def update_neighbors_batch(
        self,
        updates: list[tuple[str, str | None, str | None]],
        *,
        session: Session | None = None,
    ) -> None: ...

    # jscpd:ignore-end

    def get_rightmost_leaf_for_document(
        self, document_id: str | None
    ) -> TreeNode | None: ...
    def pin_node(self, node_id: str) -> None: ...

    # jscpd:ignore-end
    def delete_nodes(
        self,
        node_ids: Sequence[str],
        *,
        session: Session | None = None,
    ) -> None: ...

    def update_preceding_context(
        self,
        node_id: str,
        preceding_context: str | None,
    ) -> None:
        """Update the preceding_context field for a node."""
        ...

    def update_preceding_context_summary(
        self,
        node_id: str,
        summary: str | None,
    ) -> None:
        """Update the preceding_context_summary field for a node."""
        ...

    def update_embedding(
        self,
        node_id: str,
        embedding: list[float] | NDArray[np.float64] | None,
    ) -> None:
        """Update the embedding field for a node.

        The embedding is stored as packed float32 bytes for efficiency.
        """
        ...

    def update_cost(
        self,
        node_id: str,
        cost: float | None,
    ) -> None:
        """Update the cost field for a node.

        Cost is in USD for creating this node (embedding + summarization).
        """
        ...

    # Frontier tracking for contextual indexing
    def get_tree_completion_frontier(self, document_id: str | None) -> int: ...

    def get_leaves_from_span_start(
        self, document_id: str | None, span_start: int
    ) -> list[TreeNode]:
        """Get leaves with span_start >= given value, ordered by span_start.

        Used for computing the eligible span for contextual indexing gating.
        """
        ...

    def get_avg_chars_per_token(self, document_id: str | None) -> float | None:
        """Return average characters per token for leaves in a document.

        Computes SUM(span_end - span_start) / SUM(token_count) for all leaves.
        Returns None if no leaves exist yet.

        Used for estimating character positions from token budgets.
        """
        ...

    def get_nodes_by_id_prefix(
        self, document_id: str | None, id_prefix: str
    ) -> list[TreeNode]:
        """Get nodes whose ID starts with the given prefix.

        Used for CLI commands where users provide shortened node IDs.
        """
        ...

    def get_cost_stats(self, document_id: str | None) -> tuple[float, int, int, int]:
        """Get cost statistics for a document.

        Returns:
            Tuple of (total_cost, total_nodes, leaf_nodes, summary_nodes)
            where total_cost is the sum of all node costs (or 0 if no costs recorded).
        """
        ...

    # Validation
    def run_validation_queries(
        self,
        document_id: str,
        *,
        target_chunk_tokens: int | None = None,
        chunk_tolerance: float = 0.2,
    ) -> SQLValidationResult:
        """Run SQL-based validation checks, returning violations only.

        This performs fast validation by pushing checks to the database:
        - Metrics aggregations (counts, heights, etc.)
        - Leaf span gaps
        - Broken parent/child references
        - Neighbor backlink consistency
        - Level neighbor chain validation
        - Perfect binary tree structure
        - Node coordinate validation
        - Parent span union validation
        - Leaf chunk size bounds

        Args:
            document_id: Document to validate
            target_chunk_tokens: Target tokens for leaf size validation (optional)
            chunk_tolerance: Tolerance for leaf size (default 0.2 = 20%)

        Returns:
            SQLValidationResult with metrics and any violations found
        """
        ...

    # Temporal queries
    def get_leaf_at_time_position(
        self,
        document_id: str,
        time_position: float,
        position: Literal["start", "end"],
    ) -> TreeNode | None:
        """Find a leaf node at a time boundary for time→span mapping.

        This enables time-windowed queries by mapping time positions to span
        positions. The existing span-based query infrastructure can then be
        reused.

        Args:
            document_id: Document to search
            time_position: Unix timestamp (float seconds) to search for
            position: Which boundary to find:
                - "start": Earliest leaf where time_position <= leaf.time_end
                  (used as span_start for query window)
                - "end": Latest leaf where leaf.time_start <= time_position
                  (used as span_end for query window)

        Returns:
            The boundary leaf node, or None if no matching leaf exists.
            For a query window [T1, T2], call twice:
            - get_leaf_at_time_position(doc, T1, "start") → use leaf.span_start
            - get_leaf_at_time_position(doc, T2, "end") → use leaf.span_end
        """
        ...
