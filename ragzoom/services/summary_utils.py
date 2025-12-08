"""Shared helpers for summarization logic across services."""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, MutableSequence
from dataclasses import dataclass
from typing import Final, TypedDict

from ragzoom.config import IndexConfig
from ragzoom.contracts.chat_model import UsageInfo
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


SummaryMessages = MutableSequence[dict[str, str]]
SummaryCall = Callable[
    [SummaryMessages, int, str, TelemetryCollector | None],
    Awaitable[tuple[str, UsageInfo]],
]
SummaryTelemetryRecorder = Callable[[UsageInfo, int, int, float], Awaitable[None]]


class SummaryRequest(TypedDict):
    text: str
    target_tokens: int
    prev_context: str | None
    text_tokens: int | None
    parent_id: str | None
    reporter: TelemetryCollector | None


WORDS_PER_TOKEN: Final[float] = 0.75 * 0.94


@dataclass(frozen=True)
class SummaryPreparation:
    """Prepared inputs for a summarization attempt."""

    combined_text: str
    combined_tokens: int
    input_text_tokens: int
    messages: list[dict[str, str]]


@dataclass(frozen=True)
class SummaryWorkflowConfig:
    """Snapshot of configuration needed for the summary workflow."""

    summary_model: str
    use_anti_verbatim_vaccine: bool
    max_retries: int
    retry_threshold: float


def tokens_to_words(target_tokens: int) -> int:
    """Convert target token count to approximate word count."""
    return int(target_tokens * WORDS_PER_TOKEN)


def should_retry_summary(
    summary: str,
    current_tokens: int,
    target_tokens: int,
    retry_threshold: float,
) -> bool:
    """Determine whether another summarization attempt is warranted."""
    if not summary or not summary.strip():
        return True
    if target_tokens <= 0:
        return False
    if current_tokens <= target_tokens:
        return False
    deviation = (current_tokens - target_tokens) / target_tokens
    return deviation > retry_threshold


def append_retry_prompt(
    conversation: SummaryMessages,
    previous_summary: str,
    previous_tokens: int,
    target_tokens: int,
) -> None:
    """Append assistant/user messages guiding a retry to the conversation."""
    conversation.append({"role": "assistant", "content": previous_summary})
    if target_tokens > 0:
        deviation_pct = (previous_tokens - target_tokens) / target_tokens * 100
    else:
        deviation_pct = 0.0
    deviation_pct_rounded = round(abs(deviation_pct))
    larger = previous_tokens > target_tokens
    direction = "larger" if larger else "smaller"
    target_words = tokens_to_words(target_tokens)
    addendum = (
        " Use your last attempt as a starting point and aggressively prune details to hit the target words."
        if larger
        else ""
    )
    retry_prompt = (
        f"Your summary was {deviation_pct_rounded}% {direction} than the target length. "
        f"Try again, making it AT MOST {target_words} words.{addendum}"
    )
    conversation.append({"role": "user", "content": retry_prompt})


def prepare_summary_inputs(
    *,
    text: str,
    target_tokens: int,
    prev_context: str | None = None,
    text_tokens: int | None = None,
    use_anti_verbatim_vaccine: bool = False,
) -> SummaryPreparation:
    """Return prepared prompt messages and token counts for summarization."""

    combined_text = text.strip()

    if text_tokens is not None:
        combined_tokens = text_tokens
    else:
        combined_tokens = tokenizer.count_tokens(combined_text)

    input_text_tokens = combined_tokens

    target_words = tokens_to_words(target_tokens)
    instruction = (
        "You will be given a piece of content to summarize. You are to summarize ONLY the content "
        f"between the <SUMMARIZE_TEXT> tags in AT MOST {target_words} words. Use the <PRECEDING_TEXT> content as context (when provided - this may be omitted if there is no preceding context). "
        "You should be able to substitute your summary where the <SUMMARIZE_TEXT> content is and it should work just as well within the context as the original text did. The <PRECEDING_TEXT> should flow smoothly into your summary.\n\n"
        "Make your summary information-dense, covering the full temporal scope of the source material. Match the voice, tense, and tone of the original text insofar as possible. "
        "Abstract over details as necessary to fit within the word limit while preserving key events and themes.\n\n"
        "Here's the content to summarize:"
    )

    prompt_parts: list[str] = [instruction]
    if prev_context:
        prompt_parts.append(
            f"\n<PRECEDING_TEXT>\n...{prev_context.strip()}\n</PRECEDING_TEXT>"
        )
    prompt_parts.append(f"\n<SUMMARIZE_TEXT>\n{combined_text}\n</SUMMARIZE_TEXT>")

    full_prompt = "\n\n".join(prompt_parts)

    messages: list[dict[str, str]] = [
        {
            "role": "system",
            "content": "You are a precise summarizer who ONLY uses information explicitly provided in the input text. You NEVER add context or details from outside the given text.",
        },
        {"role": "user", "content": full_prompt},
    ]

    if use_anti_verbatim_vaccine:
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

    return SummaryPreparation(
        combined_text=combined_text,
        combined_tokens=combined_tokens,
        input_text_tokens=input_text_tokens,
        messages=messages,
    )


def record_passthrough_attempt(
    reporter: TelemetryCollector | None,
    parent_id: str | None,
    *,
    target_tokens: int,
    combined_tokens: int,
) -> None:
    """Record telemetry for the passthrough case where no LLM call is needed."""
    if not reporter or not parent_id:
        return

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


async def record_summary_attempt(
    reporter: TelemetryCollector | None,
    parent_id: str | None,
    *,
    usage: UsageInfo,
    target_tokens: int,
    input_text_tokens: int,
    actual_tokens: int,
    start_time: float,
    default_model: str,
) -> None:
    """Record telemetry for an LLM summary attempt, guarding against failures."""
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
            model=str(usage.get("model", default_model)),
            start_time=start_time,
            cached_tokens=cached_tokens,
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Failed to record summary telemetry: %s", exc)


def mark_accepted_attempt(
    reporter: TelemetryCollector | None,
    parent_id: str | None,
    attempt_index: int,
) -> None:
    """Mark which attempt produced the accepted summary, if telemetry is enabled."""
    if reporter and parent_id:
        reporter.mark_accepted_attempt(parent_id, attempt_index)


def is_better_summary(
    new_tokens: int,
    current_best_tokens: int,
    target_tokens: int,
) -> bool:
    """Return True if the new summary is preferable to the current best."""
    if new_tokens <= target_tokens and current_best_tokens > target_tokens:
        return True
    if new_tokens <= target_tokens and current_best_tokens <= target_tokens:
        return new_tokens > current_best_tokens
    if new_tokens > target_tokens and current_best_tokens > target_tokens:
        return new_tokens < current_best_tokens
    return False


async def retry_summary_correction(
    *,
    base_messages: list[dict[str, str]],
    initial_summary: str,
    initial_tokens: int,
    target_tokens: int,
    max_retries: int,
    retry_threshold: float,
    node_id: str,
    reporter: TelemetryCollector | None,
    call_summary: SummaryCall,
    record_attempt: SummaryTelemetryRecorder | None = None,
) -> tuple[str, int, int]:
    """Execute retry attempts to steer a summary toward the target length."""

    best_summary = initial_summary
    best_tokens = initial_tokens
    best_attempt_index = 0
    actual_retries = 0

    retry_messages: list[dict[str, str]] = list(base_messages)

    for attempt in range(1, max_retries + 1):
        try:
            retry_start = time.time()
            append_retry_prompt(
                retry_messages,
                best_summary,
                best_tokens,
                target_tokens,
            )
            retry_summary, usage = await call_summary(
                retry_messages,
                target_tokens,
                node_id,
                reporter,
            )
            actual_retries = attempt
            retry_tokens = tokenizer.count_tokens(retry_summary)

            if record_attempt is not None:
                await record_attempt(usage, best_tokens, retry_tokens, retry_start)

            if not should_retry_summary(
                retry_summary, retry_tokens, target_tokens, retry_threshold
            ):
                return retry_summary, actual_retries, actual_retries

            if is_better_summary(retry_tokens, best_tokens, target_tokens):
                best_summary = retry_summary
                best_tokens = retry_tokens
                best_attempt_index = actual_retries

        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(
                "Retry attempt %s failed for node %s: %s", attempt, node_id or "", exc
            )
            break

    if target_tokens > 0:
        deviation = abs(best_tokens - target_tokens) / target_tokens
    else:
        deviation = float("inf")

    logger.debug(
        "Summary retry complete for node %s. Best tokens: %s (deviation: %.2f%%)",
        node_id or "",
        best_tokens,
        deviation * 100,
    )

    return best_summary, actual_retries, best_attempt_index


async def run_summary_workflow(
    *,
    text: str,
    target_tokens: int,
    prev_context: str | None = None,
    text_tokens: int | None = None,
    parent_id: str | None = None,
    reporter: TelemetryCollector | None = None,
    config: SummaryWorkflowConfig,
    call_summary: SummaryCall,
) -> tuple[str, int, int]:
    """Execute the full summary workflow with retries and telemetry."""

    preparation = prepare_summary_inputs(
        text=text,
        target_tokens=target_tokens,
        prev_context=prev_context,
        text_tokens=text_tokens,
        use_anti_verbatim_vaccine=config.use_anti_verbatim_vaccine,
    )

    if preparation.combined_tokens <= target_tokens:
        record_passthrough_attempt(
            reporter,
            parent_id,
            target_tokens=target_tokens,
            combined_tokens=preparation.combined_tokens,
        )
        return (
            preparation.combined_text,
            0,
            preparation.combined_tokens,
        )

    node_id = parent_id or ""

    async def telemetry_recorder(
        usage: UsageInfo,
        input_text_tokens: int,
        actual_tokens: int,
        start_time: float,
    ) -> None:
        await record_summary_attempt(
            reporter,
            parent_id,
            usage=usage,
            target_tokens=target_tokens,
            input_text_tokens=input_text_tokens,
            actual_tokens=actual_tokens,
            start_time=start_time,
            default_model=config.summary_model,
        )

    start_time = time.time()
    summary, usage = await call_summary(
        preparation.messages,
        target_tokens,
        node_id,
        reporter,
    )
    summary_tokens = tokenizer.count_tokens(summary)

    await telemetry_recorder(
        usage,
        preparation.input_text_tokens,
        summary_tokens,
        start_time,
    )

    if not should_retry_summary(
        summary,
        summary_tokens,
        target_tokens,
        config.retry_threshold,
    ):
        mark_accepted_attempt(reporter, parent_id, 0)
        return summary, 0, summary_tokens

    if config.max_retries > 0:
        final_summary, retry_count, best_attempt_index = await retry_summary_correction(
            base_messages=preparation.messages,
            initial_summary=summary,
            initial_tokens=summary_tokens,
            target_tokens=target_tokens,
            max_retries=config.max_retries,
            retry_threshold=config.retry_threshold,
            node_id=node_id,
            reporter=reporter,
            call_summary=call_summary,
            record_attempt=telemetry_recorder,
        )
        mark_accepted_attempt(reporter, parent_id, best_attempt_index)
        return (
            final_summary,
            retry_count,
            tokenizer.count_tokens(final_summary),
        )

    mark_accepted_attempt(reporter, parent_id, 0)
    return summary, 0, summary_tokens


async def run_summary_from_config(
    *,
    index_config: IndexConfig,
    text: str,
    target_tokens: int,
    prev_context: str | None = None,
    text_tokens: int | None = None,
    parent_id: str | None = None,
    reporter: TelemetryCollector | None = None,
    call_summary: SummaryCall,
) -> tuple[str, int, int]:
    """Convenience wrapper building workflow config from IndexConfig."""

    config_snapshot = SummaryWorkflowConfig(
        summary_model=index_config.summary_model,
        use_anti_verbatim_vaccine=index_config.use_anti_verbatim_vaccine,
        max_retries=index_config.max_retries,
        retry_threshold=index_config.retry_threshold,
    )

    return await run_summary_workflow(
        text=text,
        target_tokens=target_tokens,
        prev_context=prev_context,
        text_tokens=text_tokens,
        parent_id=parent_id,
        reporter=reporter,
        config=config_snapshot,
        call_summary=call_summary,
    )


async def run_summary_request(
    *,
    index_config: IndexConfig,
    request: SummaryRequest,
    call_summary: SummaryCall,
) -> tuple[str, int, int]:
    """Execute the summary workflow using a packaged request payload."""

    return await run_summary_from_config(
        index_config=index_config,
        text=request["text"],
        target_tokens=request["target_tokens"],
        prev_context=request["prev_context"],
        text_tokens=request["text_tokens"],
        parent_id=request["parent_id"],
        reporter=request["reporter"],
        call_summary=call_summary,
    )
