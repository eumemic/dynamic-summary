"""Cost computation utilities for Anthropic cache-aware pricing.

Extracted from claude_agent_sdk backend so tests can import these
lightweight helpers without triggering the heavy SDK import chain.
"""

from __future__ import annotations

import logging
from typing import NamedTuple

from ragzoom.model_info import ModelInfo

logger = logging.getLogger(__name__)


class UsageBreakdown(NamedTuple):
    """Detailed token usage from Anthropic's ResultMessage.

    Anthropic reports three categories of input tokens, each priced differently:
    - input_tokens: tokens after the last cache breakpoint (full input price)
    - cache_creation_tokens: newly written to cache (1.25x input price)
    - cache_read_tokens: served from cache (0.1x input price, 90% discount)
    """

    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    output_tokens: int

    @property
    def total_input(self) -> int:
        """Total input tokens across all three categories."""
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens


def compute_cost(model_id: str, usage: UsageBreakdown) -> float | None:
    """Compute total cost in USD from usage breakdown and model pricing.

    Returns None if the model is not found in models.json.
    """
    try:
        info = ModelInfo()
        input_price, output_price = info.get_llm_costs(model_id)
        cache_discount = info.get_cache_discount(model_id)
        write_mult = info.get_cache_write_multiplier(model_id)
    except ValueError:
        logger.warning("Model %r not in models.json; cost not computed", model_id)
        return None

    input_cost = (usage.input_tokens / 1000) * input_price
    write_cost = (usage.cache_creation_tokens / 1000) * input_price * write_mult
    read_cost = (usage.cache_read_tokens / 1000) * input_price * (1 - cache_discount)
    output_cost = (usage.output_tokens / 1000) * output_price

    return input_cost + write_cost + read_cost + output_cost
