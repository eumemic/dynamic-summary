"""Lightweight mocks for EmbeddingModel and ChatModel.

Useful in tests to avoid patching provider SDKs and to exercise core
business logic paths (batching, retries, telemetry).
"""

from __future__ import annotations

from collections.abc import Sequence

from ragzoom.contracts.chat_model import ChatModel, ChatResult, Message
from ragzoom.contracts.embedding_model import EmbeddingModel
from ragzoom.utils.tokenization import tokenizer


class MockEmbeddingModel(EmbeddingModel):
    """Deterministic embedding model: encodes length into a small vector."""

    def __init__(self, model_id: str = "mock-embedding") -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            n = len(t)
            out.append([float(n), 1.0, 0.0, -1.0])
        return out


class MockChatModel(ChatModel):
    """Deterministic summarizer that truncates to target word budget."""

    def __init__(self, model_id: str = "mock-chat") -> None:
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    # jscpd:ignore-start - Protocol implementation must match signature
    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> ChatResult:
        # jscpd:ignore-end
        # Take the last user message content as the source
        content_src = ""
        for m in reversed(messages):
            if m["role"] == "user":
                content_src = m["content"]
                break

        # Simple heuristic: keep first 40 words (simulates target budget)
        words = content_src.split()
        summary = " ".join(words[:40]) if words else ""

        p = tokenizer.count_tokens(content_src)
        c = tokenizer.count_tokens(summary)
        from ragzoom.contracts.chat_model import UsageInfo

        usage: UsageInfo = {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": p + c,
        }
        return {"content": summary, "usage": usage}
