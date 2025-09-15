"""Summarization orchestration using a ChatModel.

Holds provider-agnostic business logic for prompt shaping, token budgeting,
retry strategy, and telemetry recording. Provider adapters remain thin.
"""

from __future__ import annotations

import logging
import time
from typing import Final

from ragzoom.config import IndexConfig, is_gpt5_model
from ragzoom.contracts.chat_model import ChatModel, ChatResult, Message, UsageInfo
from ragzoom.error_utils import preserve_exception_chain
from ragzoom.exceptions import LLMError
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


# Bias-compensated word/token ratio (see LLMService)
WORDS_PER_TOKEN: Final[float] = 0.75 * 0.94


def _tokens_to_words(target_tokens: int) -> int:
    return int(target_tokens * WORDS_PER_TOKEN)


class Summarizer:
    def __init__(self, chat_model: ChatModel, config: IndexConfig) -> None:
        self.chat_model = chat_model
        self.config = config

    async def _make_summary_call(
        self,
        messages: list[Message],
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, UsageInfo]:
        try:
            if is_gpt5_model(self.config.summary_model):
                result: ChatResult = await self.chat_model.complete(
                    messages,
                    reasoning_effort="minimal",
                )
            else:
                result = await self.chat_model.complete(messages, temperature=0.3)

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

    async def _record_summary_telemetry(
        self,
        reporter: TelemetryCollector | None,
        parent_id: str,
        usage: UsageInfo,
        target_tokens: int,
        input_text_tokens: int,
        actual_tokens: int,
        start_time: float,
    ) -> None:
        if not reporter or not parent_id:
            return

        cached_tokens = int(usage.get("cached_tokens", 0) or 0)

        try:
            reporter.record_summary_attempt_v2(
                node_id=parent_id,
                target_tokens=target_tokens,
                input_text_tokens=input_text_tokens,
                prompt_tokens=int(usage["prompt_tokens"]),
                completion_tokens=int(usage["completion_tokens"]),
                actual_tokens=actual_tokens,
                model=str(usage.get("model", self.config.summary_model)),
                start_time=start_time,
                cached_tokens=cached_tokens,
            )
        except Exception as e:
            logger.warning(f"Failed to record summary telemetry: {e}")

    def _should_retry_summary(
        self, summary: str, current_tokens: int, target_tokens: int
    ) -> bool:
        if not summary or not summary.strip():
            return True
        if target_tokens <= 0:
            return False
        # Only retry overshoots; undershoots are acceptable
        if current_tokens <= target_tokens:
            return False
        # Check if overshoot exceeds configured threshold
        deviation = (current_tokens - target_tokens) / target_tokens
        return deviation > self.config.retry_threshold

    def _is_better_summary(
        self,
        candidate_tokens: int,
        candidate_distance: int,
        best_tokens: int,
        best_distance: int,
        target_tokens: int,
    ) -> bool:
        # Prefer closer to the target; tie-break by smaller absolute tokens
        if candidate_distance != best_distance:
            return candidate_distance < best_distance
        return abs(candidate_tokens - target_tokens) < abs(best_tokens - target_tokens)

    async def _execute_retry_attempt(
        self,
        messages: list[Message],
        previous_summary: str,
        previous_tokens: int,
        target_tokens: int,
        node_id: str,
        attempt_number: int,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, UsageInfo]:
        deviation_pct = (
            (previous_tokens - target_tokens) / target_tokens * 100
            if target_tokens > 0
            else 0
        )
        deviation_pct_rounded = round(abs(deviation_pct))
        larger = previous_tokens > target_tokens
        direction = "larger" if larger else "smaller"
        target_words = _tokens_to_words(target_tokens)

        # Append the previous response and correction request
        messages.append({"role": "assistant", "content": previous_summary})
        addendum = (
            " Use your last attempt as a starting point and aggressively prune details to hit the target words."
            if larger
            else ""
        )
        retry_prompt = (
            f"Your summary was {deviation_pct_rounded}% {direction} than the target length. "
            f"Try again, making it AT MOST {target_words} words.{addendum}"
        )
        messages.append({"role": "user", "content": retry_prompt})

        summary, usage = await self._make_summary_call(
            messages, target_tokens, node_id, reporter
        )
        return summary, usage

    async def _retry_summary_correction(
        self,
        messages: list[Message],
        initial_summary: str,
        initial_tokens: int,
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, int, int]:
        best_summary = initial_summary
        best_tokens = initial_tokens
        best_distance = abs(initial_tokens - target_tokens)
        best_attempt_index = 0
        actual_retries = 0
        retry_messages = messages.copy()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                retry_start_time = time.time()
                retry_summary, retry_usage = await self._execute_retry_attempt(
                    retry_messages,
                    best_summary,
                    best_tokens,
                    target_tokens,
                    node_id,
                    attempt + 1,
                    reporter,
                )
                actual_retries = attempt
                retry_tokens = tokenizer.count_tokens(retry_summary)

                if reporter and node_id:
                    await self._record_summary_telemetry(
                        reporter=reporter,
                        parent_id=node_id,
                        usage=retry_usage,
                        target_tokens=target_tokens,
                        input_text_tokens=best_tokens,
                        actual_tokens=retry_tokens,
                        start_time=retry_start_time,
                    )

                if not self._should_retry_summary(
                    retry_summary, retry_tokens, target_tokens
                ):
                    return retry_summary, actual_retries, actual_retries

                candidate_distance = abs(retry_tokens - target_tokens)
                if self._is_better_summary(
                    retry_tokens,
                    candidate_distance,
                    best_tokens,
                    best_distance,
                    target_tokens,
                ):
                    best_summary = retry_summary
                    best_tokens = retry_tokens
                    best_distance = candidate_distance
                    best_attempt_index = actual_retries

            except Exception as e:  # pragma: no cover - defensive logging
                logger.error(f"Retry attempt {attempt} failed for node {node_id}: {e}")
                break

        return best_summary, actual_retries, best_attempt_index

    async def summarize(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        *,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
        prev_context: str | None = None,
        left_token_count: int | None = None,
        right_token_count: int | None = None,
    ) -> tuple[str, int, int]:
        combined_text = f"{left_text} {right_text}".strip()

        if left_token_count is not None and right_token_count is not None:
            combined_tokens = left_token_count + right_token_count
        else:
            combined_tokens = tokenizer.count_tokens(combined_text)
        if combined_tokens <= target_tokens:
            if reporter and parent_id:
                start_time = time.time()
                reporter.record_summary_attempt_v2(
                    node_id=parent_id,
                    target_tokens=target_tokens,
                    input_text_tokens=combined_tokens,
                    prompt_tokens=0,
                    completion_tokens=0,
                    actual_tokens=combined_tokens,
                    model="passthrough",
                    start_time=start_time,
                )
            return combined_text, 0, combined_tokens

        trimmed_prev = None
        if prev_context and self.config.preceding_context_tokens > 0:
            prev_tokens = tokenizer.encode(prev_context)
            if len(prev_tokens) > self.config.preceding_context_tokens:
                context_tokens = prev_tokens[-self.config.preceding_context_tokens :]
                trimmed_prev = tokenizer.decode(context_tokens)
            else:
                trimmed_prev = prev_context

        target_words = _tokens_to_words(target_tokens)
        instruction = (
            "You will be given a piece of content to summarize. You are to summarize ONLY the content "
            f"between the <SUMMARIZE_TEXT> tags in AT MOST {target_words} words. Use the <PRECEDING_TEXT> content as context (when provided - this may be omitted if there is no preceding context). "
            "You should be able to substitute your summary where the <SUMMARIZE_TEXT> content is and it should work just as well within the context as the original text did. The <PRECEDING_TEXT> should flow smoothly into your summary.\n\n"
            "Make your summary information-dense, covering the full temporal scope of the source material. Match the voice, tense, and tone of the original text insofar as possible. "
            "Abstract over details as necessary to fit within the word limit while preserving key events and themes.\n\n"
            "Here's the content to summarize:"
        )

        prompt_parts: list[str] = [instruction]
        if prev_context and self.config.preceding_context_tokens > 0 and trimmed_prev:
            prompt_parts.append(
                f"\n<PRECEDING_TEXT>\n...{trimmed_prev.strip()}\n</PRECEDING_TEXT>"
            )
        prompt_parts.append(f"\n<SUMMARIZE_TEXT>\n{combined_text}\n</SUMMARIZE_TEXT>")
        full_prompt = "\n\n".join(prompt_parts)

        if left_token_count is not None and right_token_count is not None:
            input_text_tokens = left_token_count + right_token_count
        else:
            input_text_tokens = tokenizer.count_tokens(left_text) + (
                tokenizer.count_tokens(right_text) if right_text else 0
            )

        messages: list[Message] = [
            {
                "role": "system",
                "content": "You are a precise summarizer who ONLY uses information explicitly provided in the input text. You NEVER add context or details from outside the given text.",
            },
            {"role": "user", "content": full_prompt},
        ]

        if self.config.use_anti_verbatim_vaccine:
            messages.append({"role": "assistant", "content": combined_text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "UNACCEPTABLE. You just returned the input text verbatim! I need you to CREATE A SUMMARY - "
                        f"extract and compress the key information to AT MOST {target_words} words. Do not copy passages directly. Try again."
                    ),
                }
            )

        try:
            start_time = time.time()
            summary, usage = await self._make_summary_call(
                messages, target_tokens, parent_id or "", reporter
            )

            summary_tokens = tokenizer.count_tokens(summary)
            if reporter and parent_id:
                await self._record_summary_telemetry(
                    reporter=reporter,
                    parent_id=parent_id,
                    usage=usage,
                    target_tokens=target_tokens,
                    input_text_tokens=input_text_tokens,
                    actual_tokens=summary_tokens,
                    start_time=start_time,
                )

            if not self._should_retry_summary(summary, summary_tokens, target_tokens):
                if reporter and parent_id:
                    reporter.mark_accepted_attempt(parent_id, 0)
                return summary, 0, summary_tokens

            if self.config.max_retries > 0:
                final_summary, retry_count, best_attempt_index = (
                    await self._retry_summary_correction(
                        messages,
                        summary,
                        summary_tokens,
                        target_tokens,
                        parent_id or "",
                        reporter,
                    )
                )
                if reporter and parent_id:
                    reporter.mark_accepted_attempt(parent_id, best_attempt_index)
                return final_summary, retry_count, tokenizer.count_tokens(final_summary)

            if reporter and parent_id:
                reporter.mark_accepted_attempt(parent_id, 0)
            return summary, 0, summary_tokens
        except Exception as e:
            llm_error = LLMError(
                operation="batch_summarize",
                model=self.config.summary_model,
                message=f"Failed to summarize text for node {parent_id}: {e}",
                node_id=parent_id,
            )
            raise preserve_exception_chain(llm_error, e)
