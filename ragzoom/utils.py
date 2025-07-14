"""Utility functions for RagZoom."""

import asyncio
import logging
from functools import wraps
from typing import Any, Callable

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter using asyncio Semaphore."""

    def __init__(self, max_concurrent: int = 10):
        """Initialize with max concurrent requests."""
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_count = 0

    async def acquire(self):
        """Acquire a slot."""
        await self.semaphore.acquire()
        self.request_count += 1
        logger.debug(f"Rate limiter: acquired slot (total requests: {self.request_count})")

    def release(self):
        """Release a slot."""
        self.semaphore.release()
        logger.debug("Rate limiter: released slot")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        self.release()


# Global rate limiter for OpenAI calls
openai_rate_limiter = RateLimiter(max_concurrent=10)


def with_rate_limit(func: Callable) -> Callable:
    """Decorator to apply rate limiting to async functions."""
    @wraps(func)
    async def wrapper(*args, **kwargs) -> Any:
        async with openai_rate_limiter:
            return await func(*args, **kwargs)
    return wrapper


def batch_process(items: list, batch_size: int) -> list:
    """Process items in batches."""
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


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
