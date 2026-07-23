"""LiteLLM adapter implementing ChatModel.

This is the SINGLE point of contact between RagZoom and ``litellm``. All other
modules speak the provider-neutral ``ChatModel`` protocol. LiteLLM normalizes
the OpenAI Chat Completions wire format and the Anthropic Messages wire format
behind one ``acompletion`` call, so the same summary path can target gpt-4o,
gpt-5.5 (OpenAI) or claude-opus-4-8 (Anthropic, optionally via a LiteLLM proxy) by changing only the
model string and optional ``api_base``/``api_key``.

litellm ships incomplete type information (see the ``litellm.*`` mypy override in
pyproject.toml). To keep ``disallow_any_explicit`` satisfied without per-line
``type: ignore`` suppressions, every value read out of a litellm response is
bound to an ``object``-typed local and narrowed with ``isinstance``/``getattr``
before it is used. No litellm-typed value escapes this module.
"""

from __future__ import annotations

import litellm

from ragzoom.contracts.chat_model import ChatModel, ChatResult, Message, UsageInfo
from ragzoom.model_info import ModelInfo


def _coerce_int(value: object, default: int = 0) -> int:
    """Coerce an untyped usage value to int without using explicit Any.

    Usage fields from litellm are dynamically typed; narrow defensively.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _extract_cached_tokens(usage: object) -> int:
    """Extract cached prompt tokens from a litellm usage object, or 0."""
    details = getattr(usage, "prompt_tokens_details", None)
    if details is None:
        return 0
    if isinstance(details, dict):
        return _coerce_int(details.get("cached_tokens", 0))
    return _coerce_int(getattr(details, "cached_tokens", 0))


class LiteLLMChatModel(ChatModel):
    """ChatModel backed by ``litellm.acompletion``.

    Args:
        model: A litellm model string (e.g. ``"gpt-4o"``, ``"gpt-5.5"``,
            ``"anthropic/claude-opus-4-8"``).
        api_base: Optional endpoint override (e.g. a proxy URL). Omitted from
            the call when ``None`` so litellm uses its own provider resolution.
        api_key: Optional API key. Omitted from the call when ``None``.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str,
        *,
        api_base: str | None = None,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._api_base = api_base
        self._api_key = api_key
        self._timeout = timeout

    @property
    def model_id(self) -> str:
        return self._model

    def _resolve_reasoning_effort(
        self, reasoning_levels: list[str], requested: str | None
    ) -> str:
        """Pick a supported reasoning level, translating unsupported requests."""
        if requested is not None and requested in reasoning_levels:
            return requested
        # Unsupported (or unspecified) level -> lowest supported level.
        return reasoning_levels[0]

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
        # Provider-neutral messages are already shaped like litellm/OpenAI
        # message dicts ({"role": ..., "content": ...}).
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "timeout": self._timeout,
        }

        if self._api_base is not None:
            kwargs["api_base"] = self._api_base
        if self._api_key is not None:
            kwargs["api_key"] = self._api_key
        if max_tokens is not None:
            kwargs["max_tokens"] = int(max_tokens)
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        # ModelInfo is the single source of truth for reasoning vs temperature.
        reasoning_levels = ModelInfo().get_reasoning_levels(self._model)
        if reasoning_levels:
            kwargs["reasoning_effort"] = self._resolve_reasoning_effort(
                reasoning_levels, reasoning_effort
            )
        elif temperature is not None:
            kwargs["temperature"] = float(temperature)

        response: object = await litellm.acompletion(**kwargs)

        content = self._extract_content(response)
        usage = self._extract_usage(response)
        return {"content": content, "usage": usage}

    def _extract_content(self, response: object) -> str:
        choices = getattr(response, "choices", None)
        if not isinstance(choices, list) or not choices:
            return ""
        message = getattr(choices[0], "message", None)
        content = getattr(message, "content", None)
        return content if isinstance(content, str) else ""

    def _extract_usage(self, response: object) -> UsageInfo:
        raw_usage = getattr(response, "usage", None)
        if raw_usage is None:
            # Minimal-usage invariant: business logic decides how to handle.
            return {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model": self._model,
            }

        usage: UsageInfo = {
            "prompt_tokens": _coerce_int(getattr(raw_usage, "prompt_tokens", 0)),
            "completion_tokens": _coerce_int(
                getattr(raw_usage, "completion_tokens", 0)
            ),
            "total_tokens": _coerce_int(getattr(raw_usage, "total_tokens", 0)),
            "model": self._model,
        }
        cached = _extract_cached_tokens(raw_usage)
        if cached > 0:
            usage["cached_tokens"] = cached
        return usage
