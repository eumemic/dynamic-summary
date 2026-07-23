"""End-to-end seam test: the summary path runs through LiteLLM.

Drives Summarizer._make_summary_call (built by LLMService) with litellm.acompletion
monkeypatched, and asserts the configured summary model + api_base reach litellm.
No network call is made.
"""

from __future__ import annotations

import pytest

from ragzoom.config import IndexConfig
from ragzoom.services.llm_service import LLMService


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 120
        self.completion_tokens = 30
        self.total_tokens = 150


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


@pytest.mark.asyncio
async def test_summary_call_uses_configured_litellm_model_and_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse("a compressed summary")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    config = IndexConfig.load(
        summary_model="anthropic/claude-opus-4-8",
        summary_api_base="https://litellm-proxy.example.com",
        summary_api_key="sk-ap-secret",
    )
    service = LLMService(config, api_key="test-key")

    content, usage = await service._summarizer._make_summary_call(
        [{"role": "user", "content": "please summarize"}],
        target_tokens=100,
        node_id="node-1",
    )

    assert content == "a compressed summary"
    assert usage["prompt_tokens"] == 120
    assert usage["completion_tokens"] == 30
    assert usage["model"] == "anthropic/claude-opus-4-8"

    # The configured model + endpoint reached litellm.
    assert captured["model"] == "anthropic/claude-opus-4-8"
    assert captured["api_base"] == "https://litellm-proxy.example.com"
    assert captured["api_key"] == "sk-ap-secret"


@pytest.mark.asyncio
async def test_opus_summary_omits_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Opus 4.8 footgun: temperature must NOT be sent (it 400s otherwise).

    The models.json reasoning_levels entry routes the call down the
    reasoning_effort branch, so summarizer.py never asks for temperature.
    """
    import litellm

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse("summary")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    config = IndexConfig.load(summary_model="claude-opus-4-8")
    service = LLMService(config, api_key="test-key")

    await service._summarizer._make_summary_call(
        [{"role": "user", "content": "please summarize"}],
        target_tokens=100,
        node_id="node-1",
    )

    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured
    assert "reasoning_effort" in captured


@pytest.mark.asyncio
async def test_non_reasoning_summary_sends_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-reasoning summary model (gpt-4o) still uses temperature=0.3."""
    import litellm

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse("summary")

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    config = IndexConfig.load(summary_model="gpt-4o")
    service = LLMService(config, api_key="test-key")

    await service._summarizer._make_summary_call(
        [{"role": "user", "content": "please summarize"}],
        target_tokens=100,
        node_id="node-1",
    )

    assert captured["temperature"] == 0.3
    assert "reasoning_effort" not in captured
