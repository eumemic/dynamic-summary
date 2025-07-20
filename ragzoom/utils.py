"""Utility functions for RagZoom."""

import logging

logger = logging.getLogger(__name__)


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
