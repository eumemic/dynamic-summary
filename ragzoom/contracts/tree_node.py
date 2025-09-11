"""Abstract TreeNode protocol used by core logic.

This protocol defines the storage-only shape for nodes that the core relies on.
Implementations (Postgres ORM rows, SQLite rows, or domain dataclasses) should
conform to this interface. Importantly, embeddings are NOT part of this
protocol — vector operations are handled by VectorIndex implementations.
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
    path: str  # binary path ("" for root)
    is_pinned: bool | int

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
    """Backend-agnostic depth computation from the binary path."""
    try:
        return int(node.get_depth())
    except Exception:
        p = getattr(node, "path", "")
        return len(p) if isinstance(p, str) else 0
