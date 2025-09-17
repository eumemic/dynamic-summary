from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DomainNode:
    """Lightweight in-memory node for indexing pipeline.

    Implements the TreeNode protocol fields used by core, plus an optional
    embedding holder used only for VectorIndex upserts. This type is NOT tied to
    any storage backend/ORM.
    """

    # Identity and scope
    id: str
    document_id: str

    # Tree relations
    parent_id: str | None = None
    left_child_id: str | None = None
    right_child_id: str | None = None

    # Positional metadata
    span_start: int = 0
    span_end: int = 0

    # Content and accounting
    text: str = ""
    token_count: int = 0
    height: int = 0
    path: str = ""
    is_pinned: bool = False

    # Neighbor relationships used by dataflow
    preceding_neighbor_id: str | None = None
    following_neighbor_id: str | None = None

    # Optional: embedding captured during dataflow; not persisted to storage
    embedding: list[float] | None = field(default=None)

    # Helpers to satisfy protocol
    def is_leaf(self) -> bool:
        return int(self.height) == 0

    def is_root(self) -> bool:
        return self.parent_id is None

    def get_depth(self) -> int:
        return len(self.path or "")

    # Additional helper used by dataflow scheduling
    def is_right_child(self) -> bool:
        p = self.path or ""
        return len(p) > 0 and p[-1] == "1"
