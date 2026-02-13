"""Tests for OpenAI backend history conversion round-trips."""

from __future__ import annotations

from ragzoom.agent.backends.openai import (
    _history_to_openai_messages,
    _openai_messages_to_history,
)
from ragzoom.agent.protocol import (
    AssistantTurn,
    MessageHistory,
    ToolCallRecord,
    ToolResultRecord,
)


class TestRoundTripSimple:
    def test_user_and_assistant(self) -> None:
        history: MessageHistory = (
            "What is 2+2?",
            AssistantTurn(text="4"),
        )
        messages = _history_to_openai_messages(history)
        restored = _openai_messages_to_history(messages)
        assert restored == history

    def test_empty_history(self) -> None:
        history: MessageHistory = ()
        messages = _history_to_openai_messages(history)
        restored = _openai_messages_to_history(messages)
        assert restored == history


class TestRoundTripWithToolCalls:
    def test_tool_call_and_result(self) -> None:
        history: MessageHistory = (
            "Search for cats",
            AssistantTurn(
                text=None,
                tool_calls=(
                    ToolCallRecord(
                        call_id="call-1",
                        tool_name="recall",
                        arguments_json='{"query": "cats"}',
                    ),
                ),
            ),
            ToolResultRecord(call_id="call-1", content="Found cats info"),
            AssistantTurn(text="Cats are great."),
        )
        messages = _history_to_openai_messages(history)
        restored = _openai_messages_to_history(messages)
        assert restored == history

    def test_assistant_with_text_and_tool_calls(self) -> None:
        history: MessageHistory = (
            "Help me",
            AssistantTurn(
                text="Let me look that up.",
                tool_calls=(
                    ToolCallRecord(
                        call_id="call-2",
                        tool_name="search",
                        arguments_json='{"q": "help"}',
                    ),
                ),
            ),
            ToolResultRecord(call_id="call-2", content="Result here"),
            AssistantTurn(text="Here's what I found."),
        )
        messages = _history_to_openai_messages(history)
        restored = _openai_messages_to_history(messages)
        assert restored == history


class TestSystemMessagesSkipped:
    def test_system_messages_filtered(self) -> None:
        from openai.types.chat import ChatCompletionMessageParam

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        history = _openai_messages_to_history(messages)
        assert len(history) == 2
        assert history[0] == "Hi"
        assert isinstance(history[1], AssistantTurn)
        assert history[1].text == "Hello!"
