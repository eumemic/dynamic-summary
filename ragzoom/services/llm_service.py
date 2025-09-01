"""LLM service for handling OpenAI API interactions."""

import logging
import time
from typing import TypedDict, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ragzoom.config import IndexConfig, SecretStr, is_gpt5_model
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


class UsageInfo(TypedDict, total=False):
    """Type definition for OpenAI API usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    cached_tokens: int  # Optional field for prompt caching


class MockUsage:
    """Mock usage object for telemetry recording."""

    def __init__(self, usage_info: UsageInfo) -> None:
        # Set basic attributes explicitly
        self.prompt_tokens = usage_info["prompt_tokens"]
        self.completion_tokens = usage_info["completion_tokens"]
        self.total_tokens = usage_info["total_tokens"]
        self.model = usage_info.get("model", "")

        # Handle prompt_tokens_details specially
        cached_tokens = usage_info.get("cached_tokens", 0)
        has_cached_tokens = cached_tokens and cached_tokens > 0
        self.prompt_tokens_details: dict[str, int] | None = (
            {"cached_tokens": cached_tokens} if has_cached_tokens else None
        )


class MockResponse:
    """Mock OpenAI response object for telemetry recording."""

    def __init__(self, usage_info: UsageInfo) -> None:
        self.usage = MockUsage(usage_info)


# Note: Removed APIParam type alias as it was too broad for OpenAI API calls
# Direct parameter passing ensures type safety with OpenAI's specific requirements


# Constants for word-based prompting bias compensation
# The 0.94 factor compensates for a systematic overshoot bias in GPT models
WORDS_PER_TOKEN = 0.75 * 0.94  # 0.705 - Bias-compensated word/token ratio


def _create_mock_response(usage_info: UsageInfo) -> MockResponse:
    """Create a mock OpenAI response object for telemetry recording."""
    return MockResponse(usage_info)


class LLMService:
    """Service for handling all LLM operations including embeddings and summarization."""

    def __init__(  # jscpd:ignore-start
        self,
        config: IndexConfig,
        api_key: str | SecretStr = "",
        max_concurrent: int = 30,
    ):
        """Initialize LLM service.

        Args:
            config: Index configuration
            api_key: OpenAI API key as SecretStr or string (if not provided, reads from OPENAI_API_KEY env)
            max_concurrent: Maximum concurrent API requests
        """  # jscpd:ignore-end
        self.config = config

        # Get API key from parameter or environment
        from ragzoom.config import ensure_secret_str

        actual_key = ensure_secret_str(api_key, "LLMService")

        self.client = AsyncOpenAI(api_key=actual_key)

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
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
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

            llm_error = LLMError(
                operation="get_embedding",
                model=self.config.embedding_model,
                message=f"Failed to get embedding: {e}",
                text_length=len(text),
            )
            raise preserve_exception_chain(llm_error, e)

    async def _get_embeddings_batch(self, texts: list[str]) -> list[list[float]]:
        """Get embeddings for multiple texts in batches to respect API limits."""
        if not texts:
            return []

        # OpenAI embeddings API has a limit of ~2048 inputs per batch
        # Use a conservative limit to avoid hitting API constraints
        max_batch_size = 1000

        if len(texts) > max_batch_size:
            logger.debug(
                f"Large batch of {len(texts)} texts - splitting into smaller batches of {max_batch_size}"
            )
            all_embeddings = []
            for i in range(0, len(texts), max_batch_size):
                batch = texts[i : i + max_batch_size]
                logger.debug(
                    f"Processing batch {i//max_batch_size + 1}/{(len(texts) + max_batch_size - 1)//max_batch_size} ({len(batch)} texts)"
                )
                batch_embeddings = await self._get_embeddings_batch(batch)
                all_embeddings.extend(batch_embeddings)
            return all_embeddings

        # Safety net: Check for empty strings that could cause API errors
        for i, text in enumerate(texts):
            if not text or not text.strip():
                logger.error(
                    f"Empty text at index {i} in embedding batch - this will cause API errors"
                )
                raise ValueError(
                    f"Empty text at index {i} in embedding batch. This should be filtered by the caller."
                )

        try:
            response = await self.client.embeddings.create(
                model=self.config.embedding_model,
                input=texts,
            )
            return [data.embedding for data in response.data]
        except Exception as e:
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

            llm_error = LLMError(
                operation="get_batch_embeddings",
                model=self.config.embedding_model,
                message=f"Failed to get batch embeddings: {e}",
                batch_size=len(texts),
            )
            raise preserve_exception_chain(llm_error, e)

    def _tokens_to_words(self, target_tokens: int) -> int:
        """Convert target token count to target word count with bias compensation."""
        return int(target_tokens * WORDS_PER_TOKEN)

    async def _make_summary_call(  # jscpd:ignore-start
        self,
        messages: list[ChatCompletionMessageParam],
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, UsageInfo]:
        """Make OpenAI API call for summarization with telemetry tracking."""  # jscpd:ignore-end
        try:
            # GPT-5 models have different parameter requirements
            if is_gpt5_model(self.config.summary_model):
                # Use reasoning_effort="minimal" (valid despite SDK type hints saying otherwise)
                response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=messages,
                    reasoning_effort="minimal",
                )
            else:
                # Only add temperature for non-GPT-5 models (GPT-5 only supports default temperature=1)
                # Use a hardcoded reasonable temperature for summaries
                response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=messages,
                    temperature=0.3,
                )

            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty response from OpenAI")

            # Extract usage information for telemetry
            if not response.usage:
                raise ValueError("No usage information in OpenAI response")

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
                prompt_tokens_details = response.usage.prompt_tokens_details
                # Handle both dict and object formats
                cached_tokens = 0
                if isinstance(prompt_tokens_details, dict):
                    cached_tokens = prompt_tokens_details.get("cached_tokens", 0) or 0
                elif hasattr(prompt_tokens_details, "cached_tokens"):
                    cached_tokens = prompt_tokens_details.cached_tokens or 0

                # Handle Mock objects in tests - they won't compare properly
                try:
                    if cached_tokens and cached_tokens > 0:
                        usage_info["cached_tokens"] = cached_tokens
                except (TypeError, AttributeError):
                    # cached_tokens might be a mock object, skip it
                    pass

            return content, cast(UsageInfo, usage_info)

        except Exception as e:
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

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
        response: MockResponse,
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

    def _should_retry_summary(
        self, summary: str, current_tokens: int, target_tokens: int
    ) -> bool:
        """Determine if a summary should be retried.

        Retries if:
        - Summary is empty or only whitespace
        - Summary overshoots target by more than the configured threshold

        Undershoots are accepted as they're already concise.

        Returns True if the summary should be retried.
        """
        # Always retry empty summaries
        if not summary or not summary.strip():
            return True

        if target_tokens <= 0:
            return False

        # Only retry overshoots - undershoots are always acceptable
        if current_tokens <= target_tokens:
            return False

        # Check if overshoot exceeds threshold
        deviation = (current_tokens - target_tokens) / target_tokens
        return deviation > self.config.retry_threshold

    def _is_better_summary(
        self,
        new_tokens: int,
        new_distance: float,
        current_best_tokens: int,
        current_best_distance: float,
        target_tokens: int,
    ) -> bool:
        """Determine if a new summary is better than the current best.

        Follows the original TreeBuilder logic:
        - Prefer summaries that are under target and closer to target
        - When current best is over target, prefer smaller summaries

        Args:
            new_tokens: Token count of new summary
            new_distance: Absolute distance from target for new summary
            current_best_tokens: Token count of current best
            current_best_distance: Absolute distance from target for current best
            target_tokens: Target token count

        Returns:
            True if new summary is better than current best
        """
        # If new is under target and current is over, new is better
        if new_tokens <= target_tokens and current_best_tokens > target_tokens:
            return True

        # If both are under target, prefer the one closer to target (larger)
        if new_tokens <= target_tokens and current_best_tokens <= target_tokens:
            return new_tokens > current_best_tokens

        # If both are over target, prefer the smaller one
        if new_tokens > target_tokens and current_best_tokens > target_tokens:
            return new_tokens < current_best_tokens

        # Current is under and new is over - keep current
        return False

    async def _execute_retry_attempt(
        self,
        messages: list[ChatCompletionMessageParam],
        previous_summary: str,
        previous_tokens: int,
        target_tokens: int,
        node_id: str,
        attempt_number: int,
        reporter: TelemetryCollector | None = None,  # jscpd:ignore-start
    ) -> tuple[str, UsageInfo]:
        """Execute a single retry attempt for summary correction."""  # jscpd:ignore-end
        # Calculate deviation details for retry prompt
        deviation_pct = (
            (previous_tokens - target_tokens) / target_tokens * 100
            if target_tokens > 0
            else 0
        )
        deviation_pct_rounded = round(abs(deviation_pct))
        larger = previous_tokens > target_tokens
        direction = "larger" if larger else "smaller"
        target_words = self._tokens_to_words(target_tokens)

        # Append the previous response and correction request to conversation
        messages.append({"role": "assistant", "content": previous_summary})

        # Build retry prompt matching original format
        addendum = (
            " Use your last attempt as a starting point and aggressively prune details to hit the target words."
            if larger
            else ""
        )
        retry_prompt = f"Your summary was {deviation_pct_rounded}% {direction} than the target length. Try again, making it AT MOST {target_words} words.{addendum}"

        messages.append({"role": "user", "content": retry_prompt})

        # Make the API call
        summary, usage_info = await self._make_summary_call(
            messages, target_tokens, node_id, reporter
        )

        return summary, usage_info

    async def _retry_summary_correction(
        self,
        messages: list[ChatCompletionMessageParam],
        initial_summary: str,
        initial_tokens: int,
        target_tokens: int,
        node_id: str,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, int, int]:
        """Attempt to correct a summary that doesn't meet target token count.

        Returns:
            tuple: (best_summary, actual_retry_count, best_token_count)
        """
        best_summary = initial_summary
        best_tokens = initial_tokens
        best_deviation = (
            abs(initial_tokens - target_tokens) / target_tokens
            if target_tokens > 0
            else float("inf")
        )
        best_attempt_index = 0  # Track which attempt is best (0 = initial)

        # Track actual retries made
        actual_retries = 0

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

                # We made an actual retry attempt
                actual_retries = attempt

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
                if not self._should_retry_summary(
                    retry_summary, retry_tokens, target_tokens
                ):
                    # This attempt is within threshold - accept it immediately
                    return retry_summary, actual_retries, actual_retries

                # This attempt is still outside threshold, but check if it's better
                retry_distance = abs(retry_tokens - target_tokens)
                best_distance = abs(best_tokens - target_tokens)

                if self._is_better_summary(
                    retry_tokens,
                    retry_distance,
                    best_tokens,
                    best_distance,
                    target_tokens,
                ):
                    best_summary = retry_summary
                    best_tokens = retry_tokens
                    best_deviation = (
                        retry_distance / target_tokens
                        if target_tokens > 0
                        else float("inf")
                    )
                    best_attempt_index = actual_retries  # Track this is the best

                # Retry attempt completed - details tracked via telemetry

            except Exception as e:
                logger.error(f"Retry attempt {attempt} failed for node {node_id}: {e}")
                break

        # Return the best attempt we found with actual retry count
        logger.debug(
            f"Summary retry complete for node {node_id}. "
            f"Best result: {best_tokens} tokens (deviation: {best_deviation:.1%})"
        )

        return best_summary, actual_retries, best_attempt_index

    async def _summarize_text(
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
        """Summarize combined text to approximately target token count.

        Returns:
            tuple: (summary, retry_count, actual_token_count)
        """
        # Combine texts
        combined_text = f"{left_text} {right_text}".strip()

        # Check if we need to summarize at all
        combined_tokens = tokenizer.count_tokens(combined_text)
        if combined_tokens <= target_tokens:
            # Skip summarization since text is already under target
            # No need to log this as it's expected behavior for small inputs
            # Record passthrough as a summary attempt for telemetry visualization
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

        # Handle preceding context if provided
        trimmed_prev = None
        if prev_context and self.config.preceding_context_tokens > 0:
            # Trim prev_context to adjacent_context_tokens
            prev_tokens = tokenizer.encode(prev_context)
            if len(prev_tokens) > self.config.preceding_context_tokens:
                context_tokens = prev_tokens[-self.config.preceding_context_tokens :]
                trimmed_prev = tokenizer.decode(context_tokens)
            else:
                trimmed_prev = prev_context

        # Build the summarization prompt inline
        target_words = self._tokens_to_words(target_tokens)

        instruction = f"""You will be given a piece of content to summarize. You are to summarize ONLY the content between the <SUMMARIZE_TEXT> tags in AT MOST {target_words} words. Use the <PRECEDING_TEXT> content as context (when provided - this may be omitted if there is no preceding context). You should be able to substitute your summary where the <SUMMARIZE_TEXT> content is and it should work just as well within the context as the original text did. The <PRECEDING_TEXT> should flow smoothly into your summary.

Make your summary information-dense, covering the full temporal scope of the source material. Match the voice, tense, and tone of the original text insofar as possible. Abstract over details as necessary to fit within the word limit while preserving key events and themes.

Here's the content to summarize:"""

        prompt_parts = [instruction]

        # Add preceding context if available
        if prev_context and self.config.preceding_context_tokens > 0 and trimmed_prev:
            prompt_parts.append(
                f"\n<PRECEDING_TEXT>\n...{trimmed_prev.strip()}\n</PRECEDING_TEXT>"
            )

        # Add the content to summarize (concatenated)
        prompt_parts.append(f"\n<SUMMARIZE_TEXT>\n{combined_text}\n</SUMMARIZE_TEXT>")

        full_prompt = "\n\n".join(prompt_parts)

        # Calculate input text tokens for metrics tracking
        if left_token_count is not None and right_token_count is not None:
            input_text_tokens = left_token_count + right_token_count
        else:
            # Fallback if token counts not provided
            input_text_tokens = tokenizer.count_tokens(left_text)
            if right_text:
                input_text_tokens += tokenizer.count_tokens(right_text)

        # Build messages in conversational format to match test expectations
        messages: list[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": "You are a precise summarizer who ONLY uses information explicitly provided in the input text. You NEVER add context or details from outside the given text.",
            },
            {"role": "user", "content": full_prompt},
        ]

        # Anti-verbatim vaccine: Insert fake conversation showing verbatim copy being rejected
        # This prevents the most common failure mode and improves consistency
        if self.config.use_anti_verbatim_vaccine:
            messages.append({"role": "assistant", "content": combined_text})
            messages.append(
                {
                    "role": "user",
                    "content": f"UNACCEPTABLE. You just returned the input text verbatim! I need you to CREATE A SUMMARY - extract and compress the key information to AT MOST {target_words} words. Do not copy passages directly. Try again.",
                }
            )

        # Make initial summary attempt
        try:
            start_time = time.time()
            summary, usage_info = await self._make_summary_call(
                messages,
                target_tokens,
                parent_id or "",
                reporter,
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
                    input_text_tokens=input_text_tokens,
                    actual_tokens=summary_tokens,
                    start_time=start_time,
                )

            # Check if retry is needed
            if not self._should_retry_summary(summary, summary_tokens, target_tokens):
                accepted_attempt = 0  # Initial attempt was accepted
                # Mark which attempt was accepted
                if reporter and parent_id:
                    reporter.mark_accepted_attempt(parent_id, accepted_attempt)
                return summary, 0, summary_tokens

            # Attempt to correct the summary
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
                # Track which attempt was accepted (like master does)
                accepted_attempt = best_attempt_index
                # Mark which attempt was accepted
                if reporter and parent_id:
                    reporter.mark_accepted_attempt(parent_id, accepted_attempt)
                return (
                    final_summary,
                    retry_count,
                    tokenizer.count_tokens(final_summary),
                )

            # No retries allowed, return initial attempt
            accepted_attempt = 0  # Initial attempt was accepted
            # Mark which attempt was accepted
            if reporter and parent_id:
                reporter.mark_accepted_attempt(parent_id, accepted_attempt)
            return summary, 0, summary_tokens

        except Exception as e:
            from ragzoom.error_utils import preserve_exception_chain
            from ragzoom.exceptions import LLMError

            llm_error = LLMError(
                operation="batch_summarize",
                model=self.config.summary_model,
                message=f"Failed to summarize text for node {parent_id}: {e}",
                node_id=parent_id,
            )
            raise preserve_exception_chain(llm_error, e)
