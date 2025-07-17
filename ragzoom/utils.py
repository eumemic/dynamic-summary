"""Utility functions for RagZoom."""

import asyncio
import logging
from collections.abc import Generator
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ragzoom.store import Store, TreeNode

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter using asyncio Semaphore."""

    def __init__(self, max_concurrent: int = 10):
        """Initialize with max concurrent requests."""
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_count = 0

    async def acquire(self) -> None:
        """Acquire a slot."""
        await self.semaphore.acquire()
        self.request_count += 1
        logger.debug(
            f"Rate limiter: acquired slot (total requests: {self.request_count})"
        )

    def release(self) -> None:
        """Release a slot."""
        self.semaphore.release()
        logger.debug("Rate limiter: released slot")

    async def __aenter__(self) -> "RateLimiter":
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        self.release()


# Global rate limiter for OpenAI calls
openai_rate_limiter = RateLimiter(max_concurrent=10)


def with_rate_limit(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to apply rate limiting to async functions."""

    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        async with openai_rate_limiter:
            return await func(*args, **kwargs)

    return wrapper


def batch_process(items: list, batch_size: int) -> Generator[list, None, None]:
    """Process items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def format_token_count(tokens: int) -> str:
    """Format token count for display.

    Args:
        tokens: Number of tokens

    Returns:
        Formatted string like "1.2k tokens" or "850 tokens"
    """
    if tokens >= 1000:
        return f"{tokens/1000:.1f}k tokens"
    return f"{tokens} tokens"


def clean_mid_delimiter(text: str) -> str:
    """Remove <<<MID>>> delimiter from text."""
    return text.replace("<<<MID>>>", "").strip()


def get_actual_node_text(
    node: "TreeNode", store: "Store", frontier_set: set[str]
) -> str:
    """
    Calculate the exact text that will be extracted for a given node
    based on which of its children are also in the frontier.

    This logic is moved from the Assembler to be reusable by the Retriever
    for more accurate budget calculations.
    """
    # If node has no mid_offset, return full text
    if not hasattr(node, "mid_offset") or node.mid_offset is None:
        return clean_mid_delimiter(node.text)

    # Get children
    left_child, right_child = store.get_children(node.id)

    left_in_frontier = left_child and left_child.id in frontier_set
    right_in_frontier = right_child and right_child.id in frontier_set

    # For parent nodes: if a child is in the frontier, it will handle its own text
    # So we should only include the OTHER child's summary from the parent
    if left_in_frontier and not right_in_frontier:
        # Left child will output its own text, parent should only output right summary
        return node.text[node.mid_offset :].strip()

    elif right_in_frontier and not left_in_frontier:
        # Right child will output its own text, parent should only output left summary
        return node.text[: node.mid_offset].strip()

    elif left_in_frontier and right_in_frontier:
        # Both children in frontier - parent should output nothing
        return ""

    # If no children are in the frontier, the parent is a leaf of the frontier
    # and should output its full summary.
    return clean_mid_delimiter(node.text)
