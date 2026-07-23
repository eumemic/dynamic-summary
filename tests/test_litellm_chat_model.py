"""Tests for the LiteLLMChatModel adapter.

All tests monkeypatch ``litellm.acompletion`` so no network call is made and the
production server is never touched. The adapter is the single point of contact
with litellm; these tests pin its translation of the ChatModel protocol knobs
(temperature vs reasoning_effort, json_mode, api_base/api_key routing) and its
usage extraction.
"""

from __future__ import annotations

import pytest

from ragzoom.adapters.litellm_chat_model import LiteLLMChatModel
from ragzoom.contracts.chat_model import Message


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakePromptTokensDetails:
    def __init__(self, cached_tokens: int) -> None:
        self.cached_tokens = cached_tokens


class _FakeUsage:
    def __init__(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cached_tokens: int | None = None,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens
        if cached_tokens is not None:
            self.prompt_tokens_details = _FakePromptTokensDetails(cached_tokens)


class _FakeResponse:
    def __init__(self, content: str | None, usage: _FakeUsage | None) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = usage


def _patch_acompletion(
    monkeypatch: pytest.MonkeyPatch,
    captured: dict[str, object],
    response: _FakeResponse,
) -> None:
    """Replace litellm.acompletion with a recorder that returns ``response``."""
    import litellm

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return response

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)


@pytest.mark.asyncio
async def test_complete_returns_chatresult_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    response = _FakeResponse("a summary", _FakeUsage(10, 5, 15))
    _patch_acompletion(monkeypatch, captured, response)

    model = LiteLLMChatModel("gpt-4o")
    messages: list[Message] = [{"role": "user", "content": "summarize this"}]
    result = await model.complete(messages, temperature=0.3)

    assert result["content"] == "a summary"
    usage = result["usage"]
    assert usage["prompt_tokens"] == 10
    assert usage["completion_tokens"] == 5
    assert usage["total_tokens"] == 15
    assert usage["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_passes_api_base_and_api_key_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("x", _FakeUsage(1, 1, 2)))

    model = LiteLLMChatModel(
        "anthropic/claude-opus-4-8",
        api_base="https://litellm-proxy.example.com",
        api_key="sk-ap-secret",
    )
    await model.complete([{"role": "user", "content": "hi"}])

    assert captured["api_base"] == "https://litellm-proxy.example.com"
    assert captured["api_key"] == "sk-ap-secret"
    assert captured["model"] == "anthropic/claude-opus-4-8"


@pytest.mark.asyncio
async def test_no_api_base_or_key_when_not_provided(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("x", _FakeUsage(1, 1, 2)))

    model = LiteLLMChatModel("gpt-4o")
    await model.complete([{"role": "user", "content": "hi"}])

    # Omit rather than pass None so litellm falls back to its own resolution.
    assert "api_base" not in captured
    assert "api_key" not in captured


@pytest.mark.asyncio
async def test_reasoning_model_uses_reasoning_effort_not_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("x", _FakeUsage(1, 1, 2)))

    # claude-opus-4-8 has reasoning_levels -> reasoning branch, no temperature.
    model = LiteLLMChatModel("claude-opus-4-8")
    await model.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.3,
        reasoning_effort="low",
    )

    assert "temperature" not in captured
    assert captured["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_non_reasoning_model_uses_temperature_not_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("x", _FakeUsage(1, 1, 2)))

    # gpt-4o has no reasoning_levels -> temperature branch.
    model = LiteLLMChatModel("gpt-4o")
    await model.complete(
        [{"role": "user", "content": "hi"}],
        temperature=0.3,
    )

    assert captured["temperature"] == 0.3
    assert "reasoning_effort" not in captured


@pytest.mark.asyncio
async def test_reasoning_effort_translated_to_supported_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("x", _FakeUsage(1, 1, 2)))

    # "minimal" is not in opus levels (low/medium/high) -> map to first level.
    model = LiteLLMChatModel("claude-opus-4-8")
    await model.complete(
        [{"role": "user", "content": "hi"}],
        reasoning_effort="minimal",
    )

    assert captured["reasoning_effort"] == "low"


@pytest.mark.asyncio
async def test_json_mode_sets_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("{}", _FakeUsage(1, 1, 2)))

    model = LiteLLMChatModel("gpt-4o")
    await model.complete(
        [{"role": "user", "content": "hi"}], json_mode=True, temperature=0.1
    )

    assert captured["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_json_mode_off_omits_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(
        monkeypatch, captured, _FakeResponse("text", _FakeUsage(1, 1, 2))
    )

    model = LiteLLMChatModel("gpt-4o")
    await model.complete([{"role": "user", "content": "hi"}], temperature=0.1)

    assert "response_format" not in captured


@pytest.mark.asyncio
async def test_max_tokens_passed_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("x", _FakeUsage(1, 1, 2)))

    model = LiteLLMChatModel("gpt-4o")
    await model.complete(
        [{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=256
    )

    assert captured["max_tokens"] == 256


@pytest.mark.asyncio
async def test_cached_tokens_extracted_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(
        monkeypatch,
        captured,
        _FakeResponse("x", _FakeUsage(100, 10, 110, cached_tokens=40)),
    )

    model = LiteLLMChatModel("gpt-4o")
    result = await model.complete([{"role": "user", "content": "hi"}], temperature=0.1)

    assert result["usage"].get("cached_tokens") == 40


@pytest.mark.asyncio
async def test_missing_usage_yields_minimal_usage_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse("x", None))

    model = LiteLLMChatModel("gpt-4o")
    result = await model.complete([{"role": "user", "content": "hi"}], temperature=0.1)

    usage = result["usage"]
    assert usage["prompt_tokens"] == 0
    assert usage["completion_tokens"] == 0
    assert usage["total_tokens"] == 0
    assert usage["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_none_content_yields_empty_string(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    _patch_acompletion(monkeypatch, captured, _FakeResponse(None, _FakeUsage(1, 0, 1)))

    model = LiteLLMChatModel("gpt-4o")
    result = await model.complete([{"role": "user", "content": "hi"}], temperature=0.1)

    assert result["content"] == ""


def test_model_id_property() -> None:
    model = LiteLLMChatModel("claude-opus-4-8")
    assert model.model_id == "claude-opus-4-8"
