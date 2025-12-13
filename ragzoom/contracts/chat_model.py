"""Chat model protocol and message/result types.

This protocol abstracts over chat-completion capable models and returns a
provider-neutral result used by business logic.
"""

from __future__ import annotations

from typing import Literal, Protocol, TypedDict, runtime_checkable


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class UsageInfo(TypedDict, total=False):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cached_tokens: int  # Optional field for prompt caching


class ChatResult(TypedDict):
    content: str
    usage: UsageInfo


@runtime_checkable
class ChatModel(Protocol):
    """Protocol for chat-capable language models.

    Business logic decides which knobs to pass (temperature, reasoning, etc.).
    Implementations forward those to the underlying provider as appropriate.
    """

    @property
    def model_id(self) -> str:  # pragma: no cover - protocol surface
        ...

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
        json_mode: bool = False,
    ) -> ChatResult:  # pragma: no cover - protocol surface
        ...
