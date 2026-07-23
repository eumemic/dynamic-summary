"""Tests for build_chat_model, the single chat-model construction site."""

from __future__ import annotations

from ragzoom.adapters.chat_model_factory import build_chat_model
from ragzoom.adapters.litellm_chat_model import LiteLLMChatModel


def test_build_with_api_base_and_key() -> None:
    model = build_chat_model(
        "claude-opus-4-8",
        api_base="https://litellm-proxy.example.com",
        api_key="sk-ap-secret",
    )
    assert isinstance(model, LiteLLMChatModel)
    assert model.model_id == "claude-opus-4-8"


def test_build_bare_model_no_endpoint() -> None:
    model = build_chat_model("gpt-4o")
    assert isinstance(model, LiteLLMChatModel)
    assert model.model_id == "gpt-4o"
