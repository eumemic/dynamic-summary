"""Select recent leaves for verbatim inclusion.

This module provides the logic for selecting the most recent leaves
(rightmost in span order) to include verbatim within a token budget.
Selected leaves are then pinned via the transient pinning mechanism.
"""

from collections.abc import Sequence
from typing import Protocol, TypeVar


class LeafNode(Protocol):
    """Protocol for leaf nodes usable in verbatim selection."""

    @property
    def id(self) -> str: ...

    @property
    def token_count(self) -> int: ...

    @property
    def span_start(self) -> int: ...

    @property
    def span_end(self) -> int: ...


T = TypeVar("T", bound=LeafNode)


def select_verbatim_leaves(
    leaves: Sequence[T],
    verbatim_budget: int,
) -> tuple[list[T], int]:
    """
    Select recent leaves (rightmost first) until budget exhausted.

    Args:
        leaves: List of leaf nodes to consider.
        verbatim_budget: Token budget for verbatim content.

    Returns:
        (selected_leaves, horizon_span_start)
        - selected_leaves: Leaves to pin, in span order (left to right)
        - horizon_span_start: Span where verbatim section begins (0 if none selected)
    """
    if not leaves or verbatim_budget <= 0:
        return [], 0

    # Sort by span_end descending (most recent first)
    sorted_leaves = sorted(leaves, key=lambda n: n.span_end, reverse=True)

    selected: list[T] = []
    budget_remaining = verbatim_budget

    for leaf in sorted_leaves:
        if leaf.token_count <= budget_remaining:
            selected.append(leaf)
            budget_remaining -= leaf.token_count

    if not selected:
        return [], 0

    # Return in span order (left to right)
    selected.sort(key=lambda n: n.span_start)
    horizon = min(leaf.span_start for leaf in selected)

    return selected, horizon
