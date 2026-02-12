"""Factory for creating model-agnostic agent backends."""

from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI

from ragzoom.agent.backends.openai import OpenAIBackend
from ragzoom.agent.protocol import BenchmarkingAgent


def is_anthropic_model(model_id: str) -> bool:
    """Check if a model ID corresponds to an Anthropic model."""
    return model_id.startswith("claude-")


def create_backend(
    model_id: str,
    openai_client: AsyncOpenAI,
    *,
    cli_path: str | Path | None = None,
) -> BenchmarkingAgent:
    """Create a BenchmarkingAgent for the given model ID.

    For Anthropic models (claude-*), uses the Claude Agent SDK.
    For all others, uses the OpenAI function-calling API.

    Args:
        cli_path: Override the Claude CLI binary path (e.g. a Docker wrapper).
    """
    if is_anthropic_model(model_id):
        from ragzoom.agent.backends.anthropic import AnthropicBackend

        return AnthropicBackend(model_id, cli_path=cli_path)
    return OpenAIBackend(openai_client, model_id)
