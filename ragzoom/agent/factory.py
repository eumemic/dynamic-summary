"""Factory for creating model-agnostic agent backends."""

from __future__ import annotations

from pathlib import Path

from openai import AsyncOpenAI

from ragzoom.agent.backends.openai import OpenAIBackend
from ragzoom.agent.protocol import BenchmarkingAgent
from ragzoom.model_info import ModelInfo


def is_claude_model(model_id: str) -> bool:
    """Check if a model ID (or alias) corresponds to a Claude model."""
    resolved = ModelInfo().resolve_model_id(model_id)
    return resolved.startswith("claude-")


def create_backend(
    model_id: str,
    openai_client: AsyncOpenAI,
    *,
    cli_path: str | Path | None = None,
    reasoning_level: str | None = None,
) -> BenchmarkingAgent:
    """Create a BenchmarkingAgent for the given model ID.

    Resolves model aliases (e.g. ``sonnet-4.5`` → ``claude-sonnet-4-5-20250929``)
    before routing to the appropriate backend.

    For Claude models, uses the Claude Agent SDK.
    For all others, uses the OpenAI function-calling API.

    Args:
        cli_path: Override the Claude CLI binary path (e.g. a Docker wrapper).
        reasoning_level: Override the reasoning effort level for reasoning models.
    """
    resolved_id = ModelInfo().resolve_model_id(model_id)
    if is_claude_model(resolved_id):
        from ragzoom.agent.backends.claude_agent_sdk import ClaudeAgentSDKBackend

        return ClaudeAgentSDKBackend(resolved_id, cli_path=cli_path)
    return OpenAIBackend(openai_client, resolved_id, reasoning_level=reasoning_level)
