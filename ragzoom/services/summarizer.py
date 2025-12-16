"""Summarization orchestration using a ChatModel.

Holds provider-agnostic business logic for prompt shaping, token budgeting,
retry strategy, and telemetry recording. Provider adapters remain thin.
"""

from __future__ import annotations

from typing import cast

from ragzoom.config import IndexConfig
from ragzoom.contracts.chat_model import ChatModel, ChatResult, Message, UsageInfo
from ragzoom.error_utils import preserve_exception_chain
from ragzoom.exceptions import LLMError
from ragzoom.model_info import ModelInfo
from ragzoom.services import summary_utils
from ragzoom.telemetry_collection import TelemetryCollector


class Summarizer:
    def __init__(self, chat_model: ChatModel, config: IndexConfig) -> None:
        self.chat_model = chat_model
        self.config = config

    async def _make_summary_call(
        self,
        messages: summary_utils.SummaryMessages,
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, UsageInfo]:
        try:
            typed_messages = cast(list[Message], messages)
            # Use reasoning_effort for models that support it, temperature otherwise.
            # The adapter handles translation of unsupported reasoning levels.
            model_info = ModelInfo()
            if model_info.get_reasoning_levels(self.config.summary_model) is not None:
                result: ChatResult = await self.chat_model.complete(
                    typed_messages,
                    reasoning_effort=self.config.summary_reasoning_level,
                )
            else:
                result = await self.chat_model.complete(typed_messages, temperature=0.3)

            content = result.get("content", "")
            if not content:
                raise ValueError("Empty response from ChatModel")

            usage = result.get("usage")
            if not usage:
                raise ValueError("No usage information in ChatModel result")

            # Ensure model id is present in usage for telemetry
            if "model" not in usage:
                usage["model"] = self.chat_model.model_id

            return content, usage
        except Exception as e:
            llm_error = LLMError(
                operation="summarize_text",
                model=self.config.summary_model,
                message=f"Failed to summarize text for node {node_id}: {e}",
                node_id=node_id,
            )
            raise preserve_exception_chain(llm_error, e)

    async def summarize(
        self,
        text: str,
        target_tokens: int,
        *,
        prev_context: str | None = None,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
        text_tokens: int | None = None,
    ) -> tuple[str, int, int]:
        request_kwargs: summary_utils.SummaryRequest = {
            "text": text,
            "target_tokens": target_tokens,
            "prev_context": prev_context,
            "text_tokens": text_tokens,
            "parent_id": parent_id,
            "reporter": reporter,
        }

        return await summary_utils.run_summary_request(
            index_config=self.config,
            request=request_kwargs,
            call_summary=self._make_summary_call,
        )
