"""Abstract TreeNode protocol used by core logic.

This protocol defines the storage-only shape for nodes that the core relies on.
Implementations (Postgres ORM rows, SQLite rows, or domain dataclasses) should
conform to this interface. Depth information must be supplied via
``get_depth()`` or a ``depth`` attribute assigned by the caller — we no longer
encode it as a persisted binary path. Vector metadata lives outside this
protocol and is handled by VectorIndex implementations.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TreeNode(Protocol):
    """Protocol for tree nodes independent of storage backend."""

    # Identity and scope
    id: str
    document_id: str | None

    # Tree relations
    parent_id: str | None
    left_child_id: str | None
    right_child_id: str | None

    # Positional metadata within the source document
    span_start: int
    span_end: int

    # Content and accounting
    text: str
    token_count: int
    height: int  # 0 for leaves, increasing toward root
    is_pinned: bool | int
    preceding_neighbor_id: str | None
    following_neighbor_id: str | None

    # Optional helpers many implementations already provide
    def is_leaf(self) -> bool: ...  # pragma: no cover - protocol signature
    def is_root(self) -> bool: ...  # pragma: no cover - protocol signature
    def get_depth(self) -> int: ...  # pragma: no cover - protocol signature


def is_leaf(node: TreeNode) -> bool:
    """Backend-agnostic leaf check, tolerant to missing helpers.

    Fallback definition: height == 0.
    """
    try:
        return bool(node.is_leaf())
    except Exception:
        return int(getattr(node, "height", 0)) == 0


def is_root(node: TreeNode) -> bool:
    """Backend-agnostic root check, tolerant to missing helpers.

    Fallback definition: parent_id is None.
    """
    try:
        return bool(node.is_root())
    except Exception:
        return getattr(node, "parent_id", None) is None


def get_depth(node: TreeNode) -> int:
    """Backend-agnostic depth computation requiring explicit depth data."""
    try:
        return int(node.get_depth())
    except Exception:
        depth_attr = getattr(node, "depth", None)
        if depth_attr is None:
            raise AttributeError(
                "TreeNode does not expose depth information; implementations must "
                "define get_depth() or provide a depth attribute."
            )
        return int(depth_attr)
