"""Protocol for node repository used by DocumentStore and services.

Defines the minimal, backend-agnostic surface area required by core code.
Implementations include Postgres and SQLite repositories.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ragzoom.contracts.tree_node import TreeNode

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
    ) -> TreeNode: ...

    def add_nodes_batch(
        self,
        nodes_data: list[dict[str, object]],
        *,
        session: Session | None = None,
    ) -> list[TreeNode]: ...

    def upsert_nodes_batch(
        self,
        nodes_data: list[dict[str, object]],
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
    def max_height_for_document(self, document_id: str | None) -> int: ...
    def get_pinned_nodes(self, depth_max: int | None = None) -> list[TreeNode]: ...
    def get_pinned_nodes_for_document(
        self, document_id: str, depth_max: int | None = None
    ) -> list[TreeNode]: ...

    # Mutations
    def update_node_access(self, node_id: str) -> None: ...
    def update_parent_references_batch(
        self, updates: list[tuple[str, str]], *, session: Session | None = None
    ) -> None: ...

    def update_neighbors_batch(
        self,
        updates: list[tuple[str, str | None, str | None]],
        *,
        session: Session | None = None,
    ) -> None: ...

    def get_rightmost_leaf_for_document(
        self, document_id: str | None
    ) -> TreeNode | None: ...
    def pin_node(self, node_id: str) -> None: ...
