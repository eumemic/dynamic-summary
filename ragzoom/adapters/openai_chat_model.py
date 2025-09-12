"""OpenAI adapter implementing ChatModel using AsyncOpenAI."""

from __future__ import annotations

from typing import Literal, cast
from typing import cast as _cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ragzoom.contracts.chat_model import ChatModel, ChatResult, Message, UsageInfo


class OpenAIChatModel(ChatModel):
    def __init__(self, client: AsyncOpenAI, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    @property
    def model_id(self) -> str:
        return self._model_id

    async def complete(
        self,
        messages: list[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        reasoning_effort: str | None = None,
    ) -> ChatResult:
        # Convert provider-neutral Message to OpenAI's param shape
        oa_messages = cast(list[ChatCompletionMessageParam], messages)

        response = await self._client.chat.completions.create(
            model=self._model_id,
            messages=oa_messages,
            temperature=float(temperature) if temperature is not None else None,
            max_tokens=int(max_tokens) if max_tokens is not None else None,
            reasoning_effort=(
                _cast(Literal["minimal", "low", "medium", "high"], reasoning_effort)
                if reasoning_effort is not None
                else None
            ),
        )

        content = response.choices[0].message.content or ""

        # Extract usage with optional cached_tokens
        if not response.usage:
            # Return minimal usage to satisfy invariant; business logic will decide how to handle
            usage: UsageInfo = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model": self._model_id,
            }
            return {"content": content, "usage": usage}

        usage_info: UsageInfo = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "model": self._model_id,
        }
        if (
            hasattr(response.usage, "prompt_tokens_details")
            and response.usage.prompt_tokens_details
        ):
            details = response.usage.prompt_tokens_details
            cached: object = 0
            if isinstance(details, dict):
                cached = details.get("cached_tokens", 0) or 0
            elif hasattr(details, "cached_tokens"):
                cached = getattr(details, "cached_tokens") or 0

            # Convert to int without using explicit Any
            if isinstance(cached, bool):
                cached_int = int(cached)
            elif isinstance(cached, int):
                cached_int = cached
            elif isinstance(cached, float):
                cached_int = int(cached)
            elif isinstance(cached, str):
                try:
                    cached_int = int(cached)
                except Exception:
                    cached_int = 0
            else:
                cached_int = 0
            if cached_int > 0:
                usage_info["cached_tokens"] = cached_int

        return {"content": content, "usage": usage_info}
