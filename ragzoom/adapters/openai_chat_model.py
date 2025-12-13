"""OpenAI adapter implementing ChatModel using AsyncOpenAI."""

from __future__ import annotations

from typing import Literal, cast
from typing import cast as _cast

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam
from openai.types.shared_params import ResponseFormatJSONObject

from ragzoom.config import is_gpt5_model
from ragzoom.contracts.chat_model import ChatModel, ChatResult, Message, UsageInfo
from ragzoom.error_handling import handle_graceful_error
from ragzoom.exceptions import LLMError


class OpenAIChatModel(ChatModel):
    def __init__(self, client: AsyncOpenAI, model_id: str) -> None:
        self._client = client
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
    ) -> ChatResult:  # jscpd:ignore-end
        # Convert provider-neutral Message to OpenAI's param shape
        oa_messages = cast(list[ChatCompletionMessageParam], messages)

        max_tokens_arg: int | NotGiven | None = (
            int(max_tokens) if max_tokens is not None else NOT_GIVEN
        )

        # Build response_format arg - use proper OpenAI type
        response_format_arg: ResponseFormatJSONObject | NotGiven = (
            ResponseFormatJSONObject(type="json_object") if json_mode else NOT_GIVEN
        )

        # GPT-5 models don't support temperature, use reasoning_effort instead
        if is_gpt5_model(self._model_id):
            reasoning_arg: Literal["minimal", "low", "medium", "high"] | NotGiven = (
                _cast(Literal["minimal", "low", "medium", "high"], reasoning_effort)
                if reasoning_effort is not None
                else "minimal"
            )
            response = await self._client.chat.completions.create(
                model=self._model_id,
                messages=oa_messages,
                max_tokens=max_tokens_arg,
                reasoning_effort=reasoning_arg,
                response_format=response_format_arg,
            )
        else:
            temp_arg: float | NotGiven | None = (
                float(temperature) if temperature is not None else NOT_GIVEN
            )
            response = await self._client.chat.completions.create(
                model=self._model_id,
                messages=oa_messages,
                temperature=temp_arg,
                max_tokens=max_tokens_arg,
                response_format=response_format_arg,
            )

        content = response.choices[0].message.content
        if not content:
            raise LLMError(
                operation="complete",
                model=self._model_id,
                message="LLM returned empty response content",
            )

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
                except Exception as exc:
                    cached_int = handle_graceful_error(
                        exc, f"Failed to parse cached_tokens '{cached}'", default=0
                    )
            else:
                cached_int = 0
            if cached_int > 0:
                usage_info["cached_tokens"] = cached_int

        return {"content": content, "usage": usage_info}
