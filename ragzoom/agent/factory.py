"""Factory for creating model-agnostic agent backends."""

from __future__ import annotations

from openai import AsyncOpenAI

from ragzoom.agent.backends.openai import OpenAIBackend
from ragzoom.agent.protocol import BenchmarkingAgent


def is_anthropic_model(model_id: str) -> bool:
    """Check if a model ID corresponds to an Anthropic model."""
    return model_id.startswith("claude-")


def create_backend(model_id: str, openai_client: AsyncOpenAI) -> BenchmarkingAgent:
    """Create a BenchmarkingAgent for the given model ID.

    For Anthropic models (claude-*), uses the Claude Agent SDK.
    For all others, uses the OpenAI function-calling API.
    """
    if is_anthropic_model(model_id):
        from ragzoom.agent.backends.anthropic import AnthropicBackend

        return AnthropicBackend(model_id)
    return OpenAIBackend(openai_client, model_id)
