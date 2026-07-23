"""The quality judge stays model-agnostic over a LiteLLM-backed ChatModel."""

from __future__ import annotations

import json

import pytest

from ragzoom.adapters.litellm_chat_model import LiteLLMChatModel
from ragzoom.evaluation.judge import evaluate_node


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeUsage:
    def __init__(self) -> None:
        self.prompt_tokens = 50
        self.completion_tokens = 20
        self.total_tokens = 70


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()


@pytest.mark.asyncio
async def test_judge_parses_dimension_scores_via_litellm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import litellm

    captured: dict[str, object] = {}
    verdict = {
        "retention": {"score": 4, "explanation": "minor omission"},
        "isolation": {"score": 5, "explanation": ""},
        "faithfulness": {"score": 5, "explanation": ""},
        "continuity": {"score": 3, "explanation": "awkward transition"},
    }

    async def fake_acompletion(**kwargs: object) -> _FakeResponse:
        captured.update(kwargs)
        return _FakeResponse(json.dumps(verdict))

    monkeypatch.setattr(litellm, "acompletion", fake_acompletion)

    judge = LiteLLMChatModel("gpt-4.1")
    scores = await evaluate_node(
        summary="a summary",
        source_text="the original source text",
        preceding_context=None,
        chat_model=judge,
    )

    assert scores["retention"].score == 4
    assert scores["continuity"].score == 3
    assert scores["isolation"].explanation == ""
    # gpt-4.1 is non-reasoning -> judge sends json_mode + temperature.
    assert captured["response_format"] == {"type": "json_object"}
    assert captured["temperature"] == 0.1
