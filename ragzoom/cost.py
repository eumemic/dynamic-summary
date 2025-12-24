"""Shared cost calculation utilities.

This module provides canonical cost calculation functions used by both:
- Telemetry analysis (post-hoc cost computation from telemetry files)
- Indexing engine (real-time cost computation during indexing)

Using the same functions ensures consistency between `ragzoom cost` CLI output
and telemetry-based cost analysis.
"""


def calculate_prompt_cost_with_cache(
    prompt_tokens: int,
    cached_tokens: int,
    price_per_1k: float,
    cache_discount: float,
) -> float:
    """Calculate prompt cost applying cache discount.

    Args:
        prompt_tokens: Total prompt tokens (includes cached tokens)
        cached_tokens: Number of tokens that were cache hits
        price_per_1k: Price per 1000 tokens
        cache_discount: Discount percentage for cached tokens (0.9 = 90% discount)

    Returns:
        Total cost in USD
    """
    uncached_tokens = prompt_tokens - cached_tokens
    return (uncached_tokens / 1000) * price_per_1k + (
        cached_tokens / 1000
    ) * price_per_1k * (1 - cache_discount)


def calculate_embedding_cost(tokens: int, price_per_1k: float) -> float:
    """Calculate embedding cost.

    Args:
        tokens: Number of tokens embedded
        price_per_1k: Price per 1000 tokens

    Returns:
        Total cost in USD
    """
    return (tokens / 1000) * price_per_1k


def calculate_completion_cost(tokens: int, price_per_1k: float) -> float:
    """Calculate completion (output) cost.

    Args:
        tokens: Number of completion tokens
        price_per_1k: Price per 1000 tokens

    Returns:
        Total cost in USD
    """
    return (tokens / 1000) * price_per_1k
