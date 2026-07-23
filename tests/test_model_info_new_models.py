"""Tests for new summary-model entries in models.json.

The LiteLLM summary refactor adds gpt-5.5 and claude-opus-4-8 so the summarizer
can route to either provider. These tests assert the entries resolve through
ModelInfo (cost, context window, reasoning levels) and that the no-fallback
guard still raises for unknown models.

Critically, claude-opus-4-8 MUST expose non-null reasoning_levels: that is what
routes the summary call down the reasoning_effort branch and prevents passing
temperature, which Opus 4.8 rejects with HTTP 400.
"""

from __future__ import annotations

import pytest

from ragzoom.model_info import ModelInfo


def test_gpt_5_5_resolves() -> None:
    info = ModelInfo()
    input_cost, output_cost = info.get_llm_costs("gpt-5.5")
    assert input_cost > 0
    assert output_cost > 0
    assert info.get_context_window("gpt-5.5") > 0
    # gpt-5.5 is a reasoning model: it must expose reasoning levels.
    assert info.get_reasoning_levels("gpt-5.5") is not None


def test_claude_opus_4_8_resolves() -> None:
    info = ModelInfo()
    input_cost, output_cost = info.get_llm_costs("claude-opus-4-8")
    assert input_cost > 0
    assert output_cost > 0
    assert info.get_context_window("claude-opus-4-8") == 200000


def test_claude_opus_4_8_has_reasoning_levels() -> None:
    """The load-bearing invariant: Opus 4.8 must route to the reasoning branch.

    Without reasoning_levels, summarizer.py passes temperature=0.3 and every
    Opus summary call 400s. A non-null reasoning_levels list is what makes
    supports_temperature() return False.
    """
    info = ModelInfo()
    levels = info.get_reasoning_levels("claude-opus-4-8")
    assert levels is not None
    assert len(levels) > 0
    assert info.supports_temperature("claude-opus-4-8") is False


def test_opus_4_8_alias_resolves() -> None:
    info = ModelInfo()
    assert info.resolve_model_id("opus-4.8") == "claude-opus-4-8"


def test_unknown_model_still_raises() -> None:
    """No-fallback guard: an unknown model id must raise, not silently default."""
    info = ModelInfo()
    with pytest.raises(ValueError):
        info.get_llm_costs("definitely-not-a-real-model-xyz")
    with pytest.raises(ValueError):
        info.get_context_window("definitely-not-a-real-model-xyz")
