"""Single construction site for ChatModel instances.

All callers (summarizer, quality judge, CLI eval commands) build chat models
through this factory so provider selection lives in exactly one place. Today
every chat model is a LiteLLMChatModel; the factory keeps that decision local.
"""

from __future__ import annotations

from ragzoom.adapters.litellm_chat_model import LiteLLMChatModel
from ragzoom.contracts.chat_model import ChatModel


def build_chat_model(
    model: str,
    *,
    api_base: str | None = None,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> ChatModel:
    """Build a ChatModel for ``model``, optionally routed via a proxy endpoint.

    Args:
        model: A litellm model string (e.g. ``"gpt-4o"``, ``"gpt-5.5"``,
            ``"anthropic/claude-opus-4-8"``).
        api_base: Optional endpoint override (proxy URL). Omitted from the
            request when ``None``.
        api_key: Optional API key. Omitted from the request when ``None``.
        timeout: Request timeout in seconds.

    Returns:
        A ChatModel ready to call ``complete``.
    """
    return LiteLLMChatModel(
        model,
        api_base=api_base,
        api_key=api_key,
        timeout=timeout,
    )
