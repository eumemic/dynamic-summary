"""LLM service for handling OpenAI API interactions."""

import asyncio
import logging
import time
from typing import Any

from openai import AsyncOpenAI

from ragzoom.config import IndexConfig
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)

# Constants for word-based prompting bias compensation
WORDS_PER_TOKEN = 0.75  # Conservative estimate for word/token ratio


def _create_mock_response(usage_info: dict[str, Any]) -> Any:
    """Create a mock OpenAI response object for telemetry recording."""

    class MockResponse:
        def __init__(self, usage_info):
            self.usage = type("Usage", (), {})()
            for key, value in usage_info.items():
                setattr(self.usage, key, value)
            # Handle prompt_tokens_details specially
            # Check if cached_tokens exists and is a real number (not a mock)
            try:
                cached_tokens = usage_info.get("cached_tokens", 0)
                has_cached_tokens = cached_tokens and cached_tokens > 0
            except (TypeError, AttributeError):
                # cached_tokens might be a mock object, treat as no caching
                has_cached_tokens = False
                cached_tokens = 0

            if has_cached_tokens:
                self.usage.prompt_tokens_details = {"cached_tokens": cached_tokens}
            else:
                self.usage.prompt_tokens_details = None

    return MockResponse(usage_info)


class LLMService:
    """Service for handling all LLM operations including embeddings and summarization."""

    def __init__(
        self,
        config: IndexConfig,
        api_key: str = "",
        max_concurrent: int = 30,
    ):
        """Initialize LLM service.

        Args:
            config: Index configuration
            api_key: OpenAI API key (if not provided, reads from OPENAI_API_KEY env)
            max_concurrent: Maximum concurrent API requests
        """
        self.config = config

        # Get API key from parameter or environment
        import os

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key required for LLMService")

        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
        async with self.semaphore:
            try:
                # Check token count before embedding
                tokens = tokenizer.encode(text)
                token_count = len(tokens)

                # Hard limit at 8000 tokens to leave margin for API overhead
                if token_count > 8000:
                    logger.error(
                        f"Text exceeds embedding token limit: {token_count} tokens "
                        f"(limit: 8000). First 200 chars: {text[:200]}..."
                    )
                    raise ValueError(
                        f"Text too large for embedding: {token_count} tokens exceeds "
                        f"limit of 8000. This is likely due to summary size growth "
                        f"at higher tree levels."
                    )

                response = await self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=text,
                    # Let OpenAI API determine dimensions - no need for hardcoded values
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"Error getting embedding: {e}")
                raise

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in a single API call."""
        if not texts:
            return []

        async with self.semaphore:
            try:
                response = await self.client.embeddings.create(
                    model=self.config.embedding_model,
                    input=texts,
                )
                return [data.embedding for data in response.data]
            except Exception as e:
                logger.error(f"Error getting batch embeddings: {e}")
                raise

    def _tokens_to_words(self, target_tokens: int) -> int:
        """Convert target token count to target word count with bias compensation."""
        return int(target_tokens * WORDS_PER_TOKEN)

    async def _make_summary_call(
        self,
        messages: list[dict[str, str]],
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Make OpenAI API call for summarization with telemetry tracking."""
        target_words = self._tokens_to_words(target_tokens)

        # Add retry instruction to the last user message
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] += (
                f" Please provide a concise summary in approximately "
                f"{target_words} words (roughly {target_tokens} tokens)."
            )

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent summaries
                    max_tokens=min(
                        target_tokens * 2, 4000
                    ),  # Allow some flexibility but cap at reasonable limit
                )

                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from OpenAI")

                # Extract usage information for telemetry
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                    "model": self.config.summary_model,
                }

                # Extract cached tokens if available (for prompt caching)
                if (
                    hasattr(response.usage, "prompt_tokens_details")
                    and response.usage.prompt_tokens_details
                ):
                    cached_tokens = response.usage.prompt_tokens_details.get(
                        "cached_tokens", 0
                    )
                    usage_info["cached_tokens"] = cached_tokens

                return content, usage_info

            except Exception as e:
                logger.error(f"Error in OpenAI API call for node {node_id}: {e}")
                raise

    async def _record_summary_telemetry(
        self,
        reporter: TelemetryCollector | None,
        parent_id: str,
        response: Any,
        target_tokens: int,
        input_text_tokens: int,
        actual_tokens: int,
        start_time: float,
    ) -> None:
        """Record telemetry for a summary attempt."""
        if not reporter or not parent_id:
            return

        # Extract cached tokens if available
        cached_tokens = 0
        if (
            hasattr(response, "usage")
            and hasattr(response.usage, "prompt_tokens_details")
            and response.usage.prompt_tokens_details
        ):
            cached_tokens = response.usage.prompt_tokens_details.get("cached_tokens", 0)

        try:
            reporter.record_summary_attempt_v2(
                node_id=parent_id,
                target_tokens=target_tokens,
                input_text_tokens=input_text_tokens,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                actual_tokens=actual_tokens,
                model=self.config.summary_model,
                start_time=start_time,
                cached_tokens=cached_tokens,
            )
        except Exception as e:
            logger.warning(f"Failed to record summary telemetry: {e}")

    def _should_retry_summary(self, current_tokens: int, target_tokens: int) -> bool:
        """Determine if a summary should be retried based on deviation from target.

        Returns True if the current summary deviates from target by more than
        the configured threshold.
        """
        if target_tokens <= 0:
            return False

        deviation = abs(current_tokens - target_tokens) / target_tokens
        return deviation > self.config.retry_threshold

    async def _execute_retry_attempt(
        self,
        messages: list[dict[str, str]],
        previous_summary: str,
        previous_tokens: int,
        target_tokens: int,
        node_id: str,
        attempt_number: int,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Execute a single retry attempt for summary correction."""
        # Calculate deviation details for retry prompt
        deviation = (
            abs(previous_tokens - target_tokens) / target_tokens
            if target_tokens > 0
            else 0
        )
        direction = "larger" if previous_tokens > target_tokens else "smaller"
        target_words = self._tokens_to_words(target_tokens)

        # Append the previous response and correction request to conversation
        messages.append({"role": "assistant", "content": previous_summary})

        retry_prompt = (
            f"Your previous summary was {previous_tokens} tokens "
            f"({deviation:.0%} {direction} than target). "
            f"Please provide a revised summary that is closer to the target of "
            f"approximately {target_words} words ({target_tokens} tokens)."
        )
        messages.append({"role": "user", "content": retry_prompt})

        # Make the API call
        summary, usage_info = await self._make_summary_call(
            messages, target_tokens, node_id, reporter
        )

        return summary, usage_info

    async def _retry_summary_correction(
        self,
        messages: list[dict[str, str]],
        initial_summary: str,
        initial_tokens: int,
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, int, int]:
        """Attempt to correct a summary that doesn't meet target token count.

        Returns:
            tuple: (best_summary, retry_count, best_token_count)
        """
        best_summary = initial_summary
        best_tokens = initial_tokens
        best_deviation = (
            abs(initial_tokens - target_tokens) / target_tokens
            if target_tokens > 0
            else float("inf")
        )

        # Create a copy of messages for retry attempts
        retry_messages = messages.copy()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Use the best attempt so far as the base for the next retry
                retry_start_time = time.time()
                retry_summary, retry_usage_info = await self._execute_retry_attempt(
                    retry_messages,
                    best_summary,
                    best_tokens,
                    target_tokens,
                    node_id,
                    attempt
                    + 1,  # attempt_number (1-indexed for initial, 2+ for retries)
                    reporter,
                )

                retry_tokens = tokenizer.count_tokens(retry_summary)

                # Record telemetry for retry attempt
                if reporter and node_id:
                    mock_response = _create_mock_response(retry_usage_info)
                    await self._record_summary_telemetry(
                        reporter=reporter,
                        parent_id=node_id,
                        response=mock_response,
                        target_tokens=target_tokens,
                        input_text_tokens=best_tokens,  # Input was the previous best attempt
                        actual_tokens=retry_tokens,
                        start_time=retry_start_time,
                    )

                # Check if this attempt is acceptable (within threshold)
                if not self._should_retry_summary(retry_tokens, target_tokens):
                    # This attempt is within threshold - accept it immediately
                    return retry_summary, attempt, retry_tokens

                # This attempt is still outside threshold, but check if it's better
                retry_deviation = (
                    abs(retry_tokens - target_tokens) / target_tokens
                    if target_tokens > 0
                    else float("inf")
                )
                if retry_deviation < best_deviation:
                    best_summary = retry_summary
                    best_tokens = retry_tokens
                    best_deviation = retry_deviation

                logger.debug(
                    f"Retry {attempt} for node {node_id}: {retry_tokens} tokens "
                    f"(deviation: {retry_deviation:.1%})"
                )

            except Exception as e:
                logger.error(f"Retry attempt {attempt} failed for node {node_id}: {e}")
                break

        # Return the best attempt we found
        logger.debug(
            f"Summary retry complete for node {node_id}. "
            f"Best result: {best_tokens} tokens (deviation: {best_deviation:.1%})"
        )

        return best_summary, self.config.max_retries, best_tokens

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, int, int]:
        """Summarize combined text to approximately target token count.

        Returns:
            tuple: (summary, retry_count, actual_token_count)
        """
        # Combine texts
        combined_text = f"{left_text} {right_text}"

        # Check if we need to summarize at all
        combined_tokens = tokenizer.count_tokens(combined_text)
        if combined_tokens <= target_tokens:
            # Text is already under target - return as-is with passthrough telemetry
            if reporter and parent_id:
                start_time = time.time()
                reporter.record_summary_attempt_v2(
                    node_id=parent_id,
                    target_tokens=target_tokens,
                    input_text_tokens=combined_tokens,
                    prompt_tokens=0,  # No LLM call made
                    completion_tokens=0,  # No LLM call made
                    actual_tokens=combined_tokens,
                    model="passthrough",  # Indicates no summarization needed - text used as-is
                    start_time=start_time,
                )
            return combined_text, 0, combined_tokens

        # Build the initial conversation
        target_words = self._tokens_to_words(target_tokens)

        # Vaccine pattern: Start with a practice round
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert at creating concise, informative summaries. "
                    "Your goal is to capture the essential information while staying "
                    "close to the target length."
                ),
            },
            {
                "role": "user",
                "content": f"Please summarize the following text in approximately {target_words} words:\n\n{combined_text}",
            },
            {
                "role": "assistant",
                "content": f"I'll provide a {target_words}-word summary of the text.",
            },
            {
                "role": "user",
                "content": (
                    "UNACCEPTABLE. I need the actual summary, not a promise to provide one. "
                    f"Please provide the actual {target_words}-word summary now."
                ),
            },
        ]

        # Make initial summary attempt
        try:
            start_time = time.time()
            summary, usage_info = await self._make_summary_call(
                messages, target_tokens, parent_id, reporter
            )

            summary_tokens = tokenizer.count_tokens(summary)

            # Record telemetry for initial attempt - create a mock response object
            if reporter and parent_id:
                mock_response = _create_mock_response(usage_info)
                await self._record_summary_telemetry(
                    reporter=reporter,
                    parent_id=parent_id,
                    response=mock_response,
                    target_tokens=target_tokens,
                    input_text_tokens=combined_tokens,
                    actual_tokens=summary_tokens,
                    start_time=start_time,
                )

            # Check if retry is needed
            if not self._should_retry_summary(summary_tokens, target_tokens):
                return summary, 0, summary_tokens

            # Attempt to correct the summary
            if self.config.max_retries > 0:
                final_summary, retry_count, final_tokens = (
                    await self._retry_summary_correction(
                        messages,
                        summary,
                        summary_tokens,
                        target_tokens,
                        parent_id,
                        reporter,
                    )
                )
                return final_summary, retry_count, final_tokens

            # No retries allowed, return initial attempt
            return summary, 0, summary_tokens

        except Exception as e:
            logger.error(f"Error summarizing text for node {parent_id}: {e}")
            raise
