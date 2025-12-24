"""Protocol for node repository used by DocumentStore and services.

Defines the minimal, backend-agnostic surface area required by core code.
Implementations include Postgres and SQLite repositories.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypedDict, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Required

# jscpd:ignore-start - Protocol declares signatures mirrored by implementations
from ragzoom.contracts.tree_node import TreeNode

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

    # Aggregations
    def count_nodes_for_document(self, document_id: str | None) -> int: ...
    def get_leaf_nodes(self) -> list[TreeNode]: ...
    def count_leaves_for_document(self, document_id: str | None) -> int: ...
    def get_recent_leaves_within_budget(
        self, document_id: str | None, token_budget: int
    ) -> list[TreeNode]: ...
    def get_recent_leaves_within_budget_before(
        self, document_id: str, token_budget: int, before_span_end: int
    ) -> list[TreeNode]: ...
    def max_height_for_document(self, document_id: str | None) -> int: ...
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
