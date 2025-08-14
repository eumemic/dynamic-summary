"""Tree building and indexing functionality for RagZoom."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, cast, overload

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from ragzoom.config import IndexConfig, is_gpt5_model
from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker
from ragzoom.splitter import TextSplitter
from ragzoom.store import Store, TreeNode
from ragzoom.telemetry_collection import TelemetryCollector

logger = logging.getLogger(__name__)


class TreeBuilder:
    """Tree builder with concurrent processing."""

    def __init__(
        self,
        config: IndexConfig,
        store: Store,
        api_key: str = "",
        max_concurrent: int = 30,
    ):
        """Initialize tree builder.

        Args:
            config: Index configuration
            store: Store instance for persistence
            api_key: OpenAI API key (if not provided, reads from OPENAI_API_KEY env)
            max_concurrent: Maximum concurrent API requests
        """
        self.config = config
        self.store = store

        # Get API key from parameter or environment
        import os

        api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OpenAI API key required for TreeBuilder")

        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.splitter = TextSplitter(config)

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return str(uuid.uuid4())

    def _validate_model_names(self) -> None:
        """Validate that configured model names are in known lists.

        This is a lightweight check that doesn't make API calls.
        Unknown models will log a warning but proceed (to support new models).
        """
        # Known valid embedding models
        valid_embedding_models = {
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        }
        if self.config.embedding_model not in valid_embedding_models:
            logger.warning(
                f"Embedding model '{self.config.embedding_model}' not in known list. "
                f"Will attempt to use it anyway. Known models: {valid_embedding_models}"
            )

        # Known valid summary models
        valid_summary_models = {
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-4",
            "gpt-3.5-turbo",
            "gpt-5-nano",
            "gpt-5-mini",
            "gpt-5",
        }
        if self.config.summary_model not in valid_summary_models:
            logger.warning(
                f"Summary model '{self.config.summary_model}' not in known list. "
                f"Will attempt to use it anyway. Known models: {valid_summary_models}"
            )

    async def _get_embedding(self, text: str) -> list[float]:
        """Get embedding for text using OpenAI."""
        async with self.semaphore:
            try:
                # Check token count before embedding
                tokens = self.splitter.tokenizer.encode(text)
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
                    # Let OpenAI API determine dimensions - no need for hardcoded values
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"Error getting batch embeddings: {e}")
                raise

    def _calculate_target_tokens(self, text: str) -> int:
        """Calculate target tokens as min of leaf_tokens or half the text size."""
        tokens = self.splitter.tokenizer.encode(text)
        half_size = len(tokens) // 2
        return min(self.config.target_chunk_tokens, half_size)

    async def _make_summary_call(
        self,
        messages: list[ChatCompletionMessageParam],
        target_tokens: int | None,
        node_info: str = "",
    ) -> tuple[str, int, Any]:
        """Make OpenAI API call and return summary, token count, and response.

        Args:
            messages: Conversation messages to send
            target_tokens: Target token count for max_tokens parameter (None for no limit)
            node_info: Optional node identifier for logging

        Returns:
            Tuple of (summary_text, token_count, raw_response)
        """
        # Build kwargs for the API call
        api_kwargs: dict[str, Any] = {
            "model": self.config.summary_model,
            "messages": messages,
        }

        # GPT-5 models have different parameter requirements
        is_gpt5 = is_gpt5_model(self.config.summary_model)

        if is_gpt5:
            # GPT-5 models need reasoning_effort="minimal" to output text instead of just reasoning
            api_kwargs["reasoning_effort"] = "minimal"
        else:
            # Only add temperature for non-GPT-5 models (GPT-5 only supports default temperature=1)
            # Use a hardcoded reasonable temperature for summaries
            api_kwargs["temperature"] = 0.3

        response = await self.client.chat.completions.create(**api_kwargs)  # type: ignore

        content = response.choices[0].message.content
        if not content:
            # Don't log here - will be logged by the caller
            raise ValueError("Empty response from LLM")

        summary = content.strip()
        if not summary:
            # Don't log here - will be logged by the caller
            raise ValueError("Empty summary after stripping")

        # Measure actual tokens
        summary_tokens = self.splitter.tokenizer.encode(summary)
        token_count = len(summary_tokens)

        return summary, token_count, response

    def _extract_cached_tokens(self, response: Any) -> int:
        """Extract cached tokens from OpenAI response.

        Args:
            response: OpenAI API response object

        Returns:
            Number of cached tokens, or 0 if not available
        """
        if not (hasattr(response, "usage") and response.usage):
            logger.debug(
                "Response missing 'usage' attribute - no cached token info available"
            )
            return 0

        if not (
            hasattr(response.usage, "prompt_tokens_details")
            and response.usage.prompt_tokens_details
        ):
            logger.debug(
                "Response usage missing 'prompt_tokens_details' - cached tokens may not be supported by this model"
            )
            return 0

        details = response.usage.prompt_tokens_details
        # Handle both dict and object cases
        if isinstance(details, dict):
            return int(details.get("cached_tokens", 0))
        elif hasattr(details, "cached_tokens"):
            return int(details.cached_tokens or 0)

        logger.debug(
            "Unexpected format for prompt_tokens_details - unable to extract cached tokens"
        )
        return 0

    async def _record_summary_telemetry(
        self,
        reporter: TelemetryCollector | None,
        parent_id: str | None,
        response: Any,
        target_tokens: int,
        input_text_tokens: int,
        actual_tokens: int,
        start_time: float,
    ) -> None:
        """Record telemetry for a summary attempt.

        Args:
            reporter: Telemetry collector instance
            parent_id: Node ID for telemetry
            response: OpenAI API response
            target_tokens: Target token count
            input_text_tokens: Input text token count
            actual_tokens: Actual summary token count
            start_time: When the API call started
        """
        if not (
            reporter and parent_id and hasattr(response, "usage") and response.usage
        ):
            return

        cached_tokens = self._extract_cached_tokens(response)

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
            logger.warning(f"Failed to record telemetry for summary attempt: {e}")

    def _should_retry_summary(self, deviation_pct: float) -> bool:
        """Check if summary should be retried based on deviation from target.

        Args:
            deviation_pct: Percentage deviation from target tokens

        Returns:
            True if retry is needed, False otherwise
        """
        return deviation_pct > self.config.retry_threshold

    def _is_better_summary(
        self,
        new_tokens: int,
        new_distance: int,
        current_best_tokens: int,
        current_best_distance: int,
        target_tokens: int,
    ) -> bool:
        """Check if a new summary attempt is better than the current best.

        Args:
            new_tokens: Token count of new attempt
            new_distance: Distance from target for new attempt
            current_best_tokens: Token count of current best
            current_best_distance: Distance from target for current best
            target_tokens: Target token count

        Returns:
            True if new attempt is better, False otherwise
        """
        # If new is under target and closer, it's better
        if new_tokens <= target_tokens and new_distance < current_best_distance:
            return True
        # If current best is over target and new is smaller, it's better
        if current_best_tokens > target_tokens and new_tokens < current_best_tokens:
            return True
        return False

    async def _execute_retry_attempt(
        self,
        messages: list[ChatCompletionMessageParam],
        target_tokens: int,
        node_info: str,
        summary: str,
        current_tokens: int | None = None,
    ) -> tuple[str, int, Any] | None:
        """Execute a single retry attempt.

        Args:
            messages: Conversation history
            target_tokens: Target token count
            node_info: Node identifier for logging
            summary: Current summary to append to conversation
            current_tokens: Already-calculated token count (avoids re-tokenization)

        Returns:
            Tuple of (new_summary, token_count, response) or None if failed
        """
        try:
            # Use provided token count or calculate if not provided
            if current_tokens is None:
                current_tokens = len(self.splitter.tokenizer.encode(summary))
            # Avoid division by zero
            if target_tokens > 0:
                deviation_pct = abs((current_tokens - target_tokens) / target_tokens)
            else:
                deviation_pct = 0.0

            # Generate inline retry prompt with runtime conditions
            retry_prompt = f"""The summary you provided deviates significantly from the target token count of {target_tokens} tokens. Your summary was {current_tokens} tokens ({deviation_pct:.1%} off target).

Please try again with a summary that:
- Is closer to exactly {target_tokens} tokens
- Maintains information density by including as much detail as possible within the token limit
- Covers the full scope of the content from start to end
- Abstracts over minor details where necessary to stay within the limit

Remember: Your goal is to maximize information preservation while hitting the target token count as closely as possible."""

            # Append current summary as assistant response
            messages.append({"role": "assistant", "content": summary})
            # Append retry instruction as user message
            messages.append({"role": "user", "content": retry_prompt})

            return await self._make_summary_call(messages, target_tokens, node_info)
        except ValueError:
            logger.warning(f"{node_info}Failed to get valid retry response")
            return None

    async def _retry_summary_correction(
        self,
        initial_summary: str,
        target_tokens: int,
        messages: list[
            ChatCompletionMessageParam
        ],  # Conversation history for continuations
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, int, int]:
        """Retry summary correction to get closer to target token count.

        Uses conversation continuations to maintain context across retries.

        Returns:
            tuple: (best_summary, actual_retry_count, best_attempt_index)
        """
        best_summary = initial_summary
        best_token_count = len(self.splitter.tokenizer.encode(initial_summary))
        best_distance_from_target = abs(best_token_count - target_tokens)
        best_attempt_index = 0

        node_info = f"[{parent_id}] " if parent_id else ""
        actual_retries = 0

        for retry_count in range(1, self.config.max_retries + 1):
            # Check if we should retry
            current_tokens = len(self.splitter.tokenizer.encode(best_summary))
            deviation_pct = abs((current_tokens - target_tokens) / target_tokens)

            if not self._should_retry_summary(deviation_pct):
                return best_summary, actual_retries, best_attempt_index

            logger.debug(
                f"{node_info}Summary deviation: {current_tokens} tokens "
                f"(target: {target_tokens}, {deviation_pct:.1%} off). "
                f"Retry {retry_count}/{self.config.max_retries}"
            )

            # Execute retry attempt (pass current_tokens to avoid re-tokenization)
            retry_start = time.time()
            result = await self._execute_retry_attempt(
                messages, target_tokens, node_info, best_summary, current_tokens
            )

            if not result:
                continue  # Failed to get valid response

            new_summary, new_token_count, retry_response = result
            actual_retries = retry_count
            new_distance = abs(new_token_count - target_tokens)

            # Track retry attempt with telemetry
            if reporter:
                await self._record_summary_telemetry(
                    reporter=reporter,
                    parent_id=parent_id,
                    response=retry_response,
                    target_tokens=target_tokens,
                    input_text_tokens=best_token_count,
                    actual_tokens=new_token_count,
                    start_time=retry_start,
                )

            # Check if this attempt is better
            if self._is_better_summary(
                new_token_count,
                new_distance,
                best_token_count,
                best_distance_from_target,
                target_tokens,
            ):
                best_summary = new_summary
                best_token_count = new_token_count
                best_distance_from_target = new_distance
                best_attempt_index = actual_retries

                logger.debug(
                    f"{node_info}Better result: {current_tokens} -> {new_token_count} tokens "
                    f"(target: {target_tokens}, distance: {new_distance})"
                )
            else:
                logger.debug(
                    f"{node_info}No improvement: {current_tokens} -> {new_token_count} tokens "
                    f"(best so far: {best_token_count} tokens)"
                )

        return best_summary, actual_retries, best_attempt_index

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        prev_context: str | None = None,
        parent_id: str | None = None,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, int, int]:
        """Summarize text using LLM.

        Returns:
            tuple: (summary, retry_count, final_token_count)
        """
        # Check if combined text is already under target
        combined_text = f"{left_text} {right_text}".strip()
        combined_tokens = self.splitter.tokenizer.encode(combined_text)
        current_token_count = len(combined_tokens)

        if current_token_count <= target_tokens:
            node_info = f"[{parent_id}] " if parent_id else ""
            logger.debug(
                f"{node_info}Combined text already under target: {current_token_count} tokens "
                f"(target: {target_tokens}). Skipping summarization."
            )

            # Record passthrough as a summary attempt for telemetry visualization
            # This helps show when nodes were processed even if they didn't need summarization
            if reporter and parent_id:
                start_time = time.time()
                reporter.record_summary_attempt_v2(
                    node_id=parent_id,
                    target_tokens=target_tokens,
                    input_text_tokens=current_token_count,
                    prompt_tokens=0,  # No LLM call made
                    completion_tokens=0,  # No LLM call made
                    actual_tokens=current_token_count,
                    model="passthrough",  # Indicates no summarization needed - text used as-is
                    start_time=start_time,
                    is_final=True,  # This is the only and final attempt
                )

            return combined_text, 0, current_token_count  # No retries needed

        # Build prompt with adjacent context (trim to avoid token explosion)
        prompt_parts: list[str] = []

        # Process adjacent context if needed
        trimmed_prev = None

        if prev_context and self.config.preceding_context_tokens > 0:
            # Trim prev_context to adjacent_context_tokens
            prev_tokens = self.splitter.tokenizer.encode(prev_context)
            if len(prev_tokens) > self.config.preceding_context_tokens:
                context_tokens = prev_tokens[-self.config.preceding_context_tokens :]
                trimmed_prev = self.splitter.tokenizer.decode(context_tokens)
            else:
                trimmed_prev = prev_context

        # Build the summarization prompt inline
        instruction = f"""You are an expert summarizer. You will be given a piece of content to summarize, embedded within the context in which it appears in a source document. You are to summarize only the content between the <SUMMARIZE_TEXT> tags in ≤{target_tokens} tokens, using the <PRECEDING_TEXT> content as context (when provided - this may be omitted if there is no preceding context). The summary should be ≤{target_tokens} tokens in total. You should be able to substitute your summary where the <SUMMARIZE_TEXT> content is and it should work just as well within the context as the original text did. The <PRECEDING_TEXT> should flow smoothly into your summary.

CRITICAL REQUIREMENTS:
- Summarize ONLY the content between the <SUMMARIZE_TEXT> and </SUMMARIZE_TEXT> tags
- The summary should be ≤{target_tokens} tokens in total
- Make the summary as information-dense as possible while filling out (but not exceeding) the token limit
- The summary should cover the full scope of the content from start to end, but abstract over minor details and omit verbal flourishes to stay within the token limit
- Focus on key events, facts, and themes ONLY from the provided text
- Do NOT include any concrete information from BEFORE the <SUMMARIZE_TEXT> tag or AFTER the </SUMMARIZE_TEXT> tag in the summary
- Do NOT complete a sentence that is cut off with information from outside the <SUMMARIZE_TEXT> block
- Do NOT infer or imagine details not present in the text
- If the text references something without explaining it, do NOT try to explain it
- IMPORTANT: Match voice and tense of the text you are summarizing! If you are summarizing a text written in the past tense, the summary MUST be in past tense
- Try to match the tone and style of the text, as if the summary were written by the same author
- Respond ONLY with your best attempt at a summary, do not break the fourth wall, say you can't summarize it, etc.

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
        input_text_tokens = len(self.splitter.tokenizer.encode(left_text)) + len(
            self.splitter.tokenizer.encode(right_text)
        )

        async with self.semaphore:
            try:
                node_info = f"[{parent_id}] " if parent_id else ""

                # Build initial messages for conversation
                # Use the original hardcoded system message for consistency with master
                messages: list[ChatCompletionMessageParam] = [
                    {
                        "role": "system",
                        "content": "You are a precise summarizer who ONLY uses information explicitly provided in the input text. You NEVER add context or details from outside the given text.",
                    },
                    {"role": "user", "content": full_prompt},
                ]

                start_time = time.time()
                try:
                    summary, current_tokens, response = await self._make_summary_call(
                        messages, target_tokens, node_info=node_info
                    )
                except ValueError:
                    logger.warning(
                        f"{node_info}LLM returned empty response, using original text as fallback"
                    )
                    summary = combined_text
                    current_tokens = len(self.splitter.tokenizer.encode(summary))
                    response = None

                # Check if summary needs correction
                deviation_pct = (
                    abs((current_tokens - target_tokens) / target_tokens)
                    if response
                    else 0
                )

                # Track the initial attempt with telemetry
                if response:
                    await self._record_summary_telemetry(
                        reporter=reporter,
                        parent_id=parent_id,
                        response=response,
                        target_tokens=target_tokens,
                        input_text_tokens=input_text_tokens,
                        actual_tokens=current_tokens,
                        start_time=start_time,
                    )

                # Use retry correction if deviation exceeds threshold
                retry_count = 0
                accepted_attempt = 0  # Default to first attempt

                if deviation_pct > self.config.retry_threshold:
                    summary, retry_count, best_attempt_index = (
                        await self._retry_summary_correction(
                            summary,
                            target_tokens,
                            messages,
                            parent_id,
                            reporter,
                        )
                    )
                    # The accepted attempt is the initial (0) plus the best retry index
                    accepted_attempt = best_attempt_index

                # Log final summary statistics consistently
                final_tokens = len(self.splitter.tokenizer.encode(summary))
                utilization_pct = (final_tokens / target_tokens) * 100
                deviation = final_tokens - target_tokens
                deviation_str = f"+{deviation}" if deviation >= 0 else str(deviation)

                logger.debug(
                    f"{node_info}Summary: {final_tokens} tokens "
                    f"(target: {target_tokens}, {deviation_str} tokens, {utilization_pct:.0f}% utilization)"
                )

                # Mark which attempt was accepted
                if reporter and parent_id:
                    try:
                        reporter.mark_accepted_attempt(parent_id, accepted_attempt)
                    except Exception as e:
                        logger.warning(
                            f"Failed to mark accepted telemetry attempt: {e}"
                        )

            except Exception as e:
                logger.error(f"Error summarizing text: {e}")
                raise

        return summary, retry_count, final_tokens

    # Methods to append to index.py

    def _update_parent_reference(self, node_id: str, parent_id: str) -> None:
        """Update a node's parent reference."""
        with self.store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            node = session.query(TreeNode).filter_by(id=node_id).first()
            if node:
                node.parent_id = parent_id
                session.commit()

                # Invalidate the cache entry for this node since we've updated it
                if node_id in self.store.node_cache:
                    del self.store.node_cache[node_id]
                    if node_id in self.store.cache_order:
                        self.store.cache_order.remove(node_id)

    @overload
    async def _add_document_impl(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        reporter: None = None,
    ) -> str: ...

    @overload
    async def _add_document_impl(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        reporter: TelemetryCollector = ...,
    ) -> tuple[str, dict]: ...

    async def _add_document_impl(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
        reporter: TelemetryCollector | None = None,
    ) -> str | tuple[str, dict]:
        """Add a document to the tree, creating leaf nodes.

        Returns:
            If reporter is None: document_id
            If reporter is provided: (document_id, metrics)
        """
        # Validate model names to warn about potential issues
        self._validate_model_names()

        # Compute content hash
        content_hash = self.store.compute_content_hash(text)

        # Check if document already exists
        existing_doc = None
        if file_path:
            existing_doc = self.store.get_document_by_path(file_path)
            if existing_doc:
                # Check if content changed
                if existing_doc.content_hash == content_hash:
                    logger.info(
                        f"Document at {file_path} unchanged, skipping re-indexing"
                    )
                    return existing_doc.id
                else:
                    logger.info(f"Document at {file_path} has changed, re-indexing...")
                    # Delete old nodes
                    deleted = self.store.delete_document_nodes(existing_doc.id)
                    logger.info(f"Deleted {deleted} old nodes")
                    document_id = existing_doc.id

        if not document_id:
            if file_path:
                # Use filename (without path) as document_id
                from pathlib import Path

                document_id = Path(file_path).name
            else:
                document_id = self._generate_node_id()

        # Split into chunks
        chunks = self.splitter.split_text(text)

        # Create progress tracker early so we can use it for logging
        # When progress bar is active, we suppress info logs to avoid disrupting the display
        progress = (
            GlobalProgressTracker(len(chunks), show_progress) if show_progress else None
        )

        # Log only when progress bar is not active to avoid display issues
        if not show_progress:
            logger.info("Splitting document into chunks...")
            logger.info(f"Split document into {len(chunks)} chunks")

        # Early validation: Check chunk sizes immediately after splitting
        from ragzoom.validate import validate, validate_chunk_sizes

        # Create simple objects with just the fields needed for validation
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_obj = type("ChunkObj", (), {"text": chunk, "id": f"chunk_{i}"})()
            chunk_objects.append(chunk_obj)

        validate(
            lambda: validate_chunk_sizes(
                chunk_objects, self.config.target_chunk_tokens
            ),
            "early chunk size validation",
        )

        # Create async wrapper for progress (tracker already created above)
        async_progress = AsyncProgressWrapper(progress) if progress else None

        # Track overall start time for cumulative elapsed time
        overall_start_time = time.time()

        try:
            # Create leaf nodes with batch embeddings
            if not show_progress and len(chunks) > 100:
                logger.info("Preparing chunk data...")

            leaf_ids: list[str] = []
            chunk_data: list[dict[str, Any]] = []

            # Prepare all chunk data with character positions
            # Now that splitter handles whitespace gaps, positioning is straightforward
            current_pos = 0
            for i, chunk in enumerate(chunks):
                node_id = self._generate_node_id()

                # Chunks now have complete coverage with no gaps
                chunk_start = current_pos
                chunk_end = chunk_start + len(chunk)

                # Verify this chunk matches the original text
                if text[chunk_start:chunk_end] != chunk:
                    # This should not happen with the fixed splitter, but provide fallback
                    logger.warning(
                        f"Chunk {i} position mismatch, using find() fallback"
                    )
                    chunk_start = text.find(chunk, current_pos)
                    if chunk_start == -1:
                        logger.error(f"Could not find chunk {i} in text")
                        chunk_start = current_pos
                    chunk_end = chunk_start + len(chunk)

                chunk_data.append(
                    {
                        "id": node_id,
                        "text": chunk,
                        "span_start": chunk_start,
                        "span_end": chunk_end,
                    }
                )

                # Track node creation for telemetry
                if reporter:
                    reporter.track_node_created(
                        node_id=node_id,
                        height=0,  # Leaves have height 0
                        span=(chunk_start, chunk_end),
                    )
                leaf_ids.append(node_id)

                # Track chunk creation
                if reporter:
                    try:
                        chunk_tokens = len(self.splitter.tokenizer.encode(chunk))
                        reporter.record_chunk_created(node_id, chunk_tokens)
                    except Exception as e:
                        logger.warning(
                            f"Failed to record telemetry for chunk creation: {e}"
                        )

                current_pos = chunk_end

            # Early validation: Check document coverage before processing embeddings
            from ragzoom.validate import validate, validate_document_coverage

            # Create node objects for validation using actual chunk data
            leaf_nodes_for_validation = []
            for data in chunk_data:
                node_obj = type(
                    "Node",
                    (),
                    {
                        "id": data["id"],
                        "span_start": data["span_start"],
                        "span_end": data["span_end"],
                        "text": data["text"],
                    },
                )()
                leaf_nodes_for_validation.append(node_obj)

            validate(
                lambda: validate_document_coverage(text, leaf_nodes_for_validation),
                "early document coverage check",
            )

            # Get embeddings in batches
            batch_size = self.config.embedding_batch_size
            all_embeddings = []

            for i in range(0, len(chunks), batch_size):
                batch_texts = [
                    cast(str, d["text"]) for d in chunk_data[i : i + batch_size]
                ]
                batch_end = min(i + batch_size, len(chunks))

                # Show which batch we're processing with cumulative elapsed time
                if not show_progress:
                    elapsed = time.time() - overall_start_time
                    mins, secs = divmod(int(elapsed), 60)
                    logger.info(
                        f"Processing embedding batch: chunks {i+1}-{batch_end} of {len(chunks)} [{mins}m {secs}s elapsed]"
                    )

                # Track embedding call with node-level detail
                if reporter:
                    node_embeddings = []
                    for j in range(i, batch_end):
                        node_id = chunk_data[j]["id"]
                        text = chunk_data[j]["text"]
                        token_count = len(self.splitter.tokenizer.encode(text))
                        node_embeddings.append((node_id, token_count))

                    start_time = time.time()

                batch_embeddings = await self._get_embeddings_batch(batch_texts)

                if reporter:
                    reporter.record_embedding_call_v2(
                        node_embeddings=node_embeddings,
                        batch_size=len(batch_texts),
                        model=self.config.embedding_model,
                        start_time=start_time,
                    )
                all_embeddings.extend(batch_embeddings)

                # Update progress for embeddings
                if async_progress:
                    await async_progress.update(len(batch_texts))

            # Prepare all leaf nodes for batch insertion
            leaf_nodes_data = []
            preceding_leaf_id = None  # Track preceding leaf for document order

            for i, (data, embedding) in enumerate(zip(chunk_data, all_embeddings)):
                text = cast(str, data["text"])
                # Count tokens for leaf nodes using tiktoken
                token_count = len(self.splitter.tokenizer.encode(text))

                leaf_nodes_data.append(
                    {
                        "node_id": cast(str, data["id"]),
                        "text": text,
                        "embedding": embedding,
                        "span_start": cast(int, data["span_start"]),
                        "span_end": cast(int, data["span_end"]),
                        "document_id": document_id,
                        "token_count": token_count,
                        "preceding_neighbor_id": preceding_leaf_id,
                    }
                )

                # Update preceding ID for next iteration
                preceding_leaf_id = cast(str, data["id"])

            # Batch insert all leaf nodes at once
            if leaf_nodes_data:
                self.store.add_nodes_batch(leaf_nodes_data)

            # Add document record
            if not existing_doc:
                self.store.add_document(
                    document_id,
                    file_path,
                    content_hash,
                    len(chunks),
                    self.config.embedding_model,
                    self.config.summary_model,
                )
            else:
                # Update existing document
                with self.store.SessionLocal() as session:
                    from ragzoom.store import Document

                    doc = session.query(Document).filter_by(id=document_id).first()
                    if doc:
                        doc.content_hash = content_hash
                        doc.chunk_count = len(chunks)
                        doc.indexed_at = datetime.utcnow()
                        doc.embedding_model = self.config.embedding_model
                        doc.summary_model = self.config.summary_model
                        session.commit()

            # Build tree from leaves
            root_id = await self._build_tree_from_leaves(
                leaf_ids,
                chunks,
                document_id,
                async_progress,
                overall_start_time,
                reporter,
            )

            # Final completion logging with total elapsed time
            if root_id:
                total_elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(total_elapsed), 60)
                if not show_progress:
                    logger.info(
                        f"Document indexed successfully: {document_id} [{mins}m {secs}s total elapsed]"
                    )

            # Finalize telemetry if collector was used
            if reporter:
                telemetry = reporter.finalize()
                return document_id, telemetry

            return document_id
        finally:
            # Always close progress
            if progress:
                progress.close()

    def add_document(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
    ) -> str:
        """Sync wrapper for add_document."""
        return asyncio.run(
            self.add_document_async(text, document_id, file_path, show_progress)
        )

    def add_document_with_telemetry(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = False,
    ) -> tuple[str, dict]:
        """Add document and return telemetry data. Used for benchmarking.

        This is a convenience method that creates a TelemetryCollector internally
        and returns the collected telemetry data. For production use, add_document() is preferred
        as it doesn't have the overhead of telemetry collection.

        The dual-method pattern ensures:
        - Normal indexing (add_document) has zero telemetry overhead
        - Benchmarking gets detailed telemetry without modifying core logic
        - Internal implementation (_add_document_impl) remains flexible

        Returns:
            Tuple of (document_id, telemetry_dict)
        """
        # Create collector internally with config for pricing
        source_tokens = len(self.splitter.tokenizer.encode(text))
        collector = TelemetryCollector(
            document_id or "benchmark",
            source_tokens,
            self.config,
            document_path=file_path,
        )

        # Run indexing with collector - will return (doc_id, telemetry)
        result = asyncio.run(
            self._add_document_impl(
                text, document_id, file_path, show_progress, collector
            )
        )

        # Extract tuple returned when collector is provided
        # Type checker knows result is a tuple because we passed a collector
        doc_id, telemetry = result
        return doc_id, telemetry

    async def add_document_async(
        self,
        text: str,
        document_id: str | None = None,
        file_path: str | None = None,
        show_progress: bool = True,
    ) -> str:
        """Async version of add_document - called by sync wrapper."""
        result = await self._add_document_impl(
            text, document_id, file_path, show_progress
        )
        # Type checker knows result is a string when no reporter is provided
        return result

    async def _process_node_pair(
        self,
        left_id: str,
        left_text: str,
        right_id: str | None,
        right_text: str | None,
        prev_context: str | None,
        document_id: str | None,
        reporter: TelemetryCollector | None = None,
        left_node: TreeNode | None = None,  # Pre-fetched node data
        right_node: TreeNode | None = None,  # Pre-fetched node data
    ) -> dict[str, Any]:
        """Process a single node pair - generate summary and embedding.

        Returns:
            Dictionary containing node data and parent updates to be applied later
        """
        parent_id = self._generate_node_id()

        # Use pre-fetched nodes if provided, otherwise fetch them
        if left_node is None:
            left_node = self.store.get_node(left_id)
        if right_id and right_node is None:
            right_node = self.store.get_node(right_id)

        if not left_node:
            logger.error(f"Failed to retrieve left child node: {left_id}")
            raise ValueError("Left child node not found in store")
        if right_id and not right_node:
            logger.error(f"Failed to retrieve right child node: {right_id}")
            raise ValueError("Right child node not found in store")

        # Track parent node creation with span from children
        if reporter:
            if right_node:
                parent_span = (left_node.span_start, right_node.span_end)
            else:
                parent_span = (left_node.span_start, left_node.span_end)
            reporter.track_node_created(
                node_id=parent_id,
                height=reporter._current_height + 1,  # Parent is one level higher
                span=parent_span,
            )

        # Use consistent token budget for all heights
        # Target tokens for the summary (guidance for LLM, not hard limit)
        target_tokens = self.config.target_chunk_tokens

        # Generate summary (async) with retry mechanism support
        summary, retry_count, token_count = await self._summarize_text(
            left_text,
            right_text or "",  # Pass empty string if no right text
            target_tokens,
            prev_context,
            parent_id,
            reporter=reporter,
        )

        # Embedding will be generated in batch after all summaries are collected
        # This avoids 183 individual API calls for a typical level

        # Return data to be stored later in batch
        return {
            "node_data": {
                "node_id": parent_id,
                "text": summary,
                "embedding": None,  # Will be filled in after batch generation
                "span_start": left_node.span_start,
                "span_end": right_node.span_end if right_node else left_node.span_end,
                "left_child_id": left_id,
                "right_child_id": right_id,  # Can be None
                "document_id": document_id,
                "token_count": token_count,
            },
            "parent_updates": [
                (left_id, parent_id),
                (right_id, parent_id) if right_id else None,
            ],
            "parent_id": parent_id,
            "summary": summary,
            "token_count": token_count,  # Pass token count for telemetry
            # Store validation data for later
            "validation_data": {
                "left_span_start": left_node.span_start,
                "right_span_end": right_node.span_end if right_node else None,
            },
        }

    async def _build_tree_from_leaves(
        self,
        leaf_ids: list[str],
        leaf_texts: list[str],
        document_id: str | None = None,
        progress: AsyncProgressWrapper | None = None,
        overall_start_time: float | None = None,
        reporter: TelemetryCollector | None = None,
    ) -> str:
        """Build tree bottom-up from leaf nodes with concurrent processing."""
        current_level_ids = leaf_ids
        current_level_texts = leaf_texts

        # Calculate total tree height (distance from root to furthest leaf)
        # Note: This is used for progress tracking estimation

        # Track leaf level
        if reporter:
            try:
                reporter.record_tree_height_complete(0, len(leaf_ids))
            except Exception as e:
                logger.warning(f"Failed to record telemetry for tree height: {e}")

        current_height = 1  # Track height for logging (leaves are at height 0)
        while len(current_level_ids) > 1:
            next_level_ids: list[str] = []
            next_level_texts: list[str] = []
            # Note: current_height will be incremented after processing this height

            # Pre-fetch all nodes for this level to avoid individual DB queries in tasks
            all_nodes = self.store.get_nodes(current_level_ids)
            nodes_by_id = {node.id: node for node in all_nodes}

            # Process pairs concurrently
            tasks = []
            pair_info: list[tuple[int, int | None]] = []

            # Process all nodes in pairs, with the last one having no right child if odd
            i = 0
            while i < len(current_level_ids):
                left_id = current_level_ids[i]
                left_text = current_level_texts[i]

                # Check if we have a right node
                if i + 1 < len(current_level_ids):
                    right_id = current_level_ids[i + 1]
                    right_text = current_level_texts[i + 1]
                    pair_info.append((i, i + 1))
                    i += 2  # Move to next pair
                else:
                    # Odd node - no right child
                    right_id = None
                    right_text = None
                    pair_info.append((i, None))
                    i += 1  # This was the last node

                # Get adjacent context
                prev_context = None
                if pair_info[-1][0] > 0:  # Use the left index from the current pair
                    prev_context, _ = self.splitter.get_adjacent_context(
                        current_level_texts, pair_info[-1][0] - 1
                    )

                # Get pre-fetched nodes
                left_node = nodes_by_id.get(left_id)
                right_node = nodes_by_id.get(right_id) if right_id else None

                # Create async task with pre-fetched nodes
                task = self._process_node_pair(
                    left_id,
                    left_text,
                    right_id,
                    right_text,
                    prev_context,
                    document_id,
                    reporter,
                    left_node=left_node,
                    right_node=right_node,
                )
                tasks.append(task)

            # Process all pairs concurrently
            if tasks:
                # Log tree building progress only when no progress bar
                if not (
                    progress and progress.tracker and progress.tracker.show_progress
                ):
                    if overall_start_time:
                        elapsed = time.time() - overall_start_time
                        mins, secs = divmod(int(elapsed), 60)
                        logger.info(
                            f"Building tree height {current_height}: processing {len(tasks)} node pairs [{mins}m {secs}s elapsed]"
                        )
                    else:
                        logger.info(
                            f"Building tree height {current_height}: processing {len(tasks)} node pairs"
                        )

                # Track completion count
                completed_count = 0

                # Wrap each task to update progress when it completes
                async def track_progress(task: Any, task_index: int) -> Any:
                    nonlocal completed_count
                    result = await task

                    # Update progress immediately when this pair completes
                    # For odd nodes (single child), only update by 1
                    if progress:
                        # Check if this is an odd node (has None for right child)
                        # task_index comes from enumerate(tasks) and corresponds to the position
                        # in both the tasks list and pair_info list (created in the same loop).
                        # Even though tasks complete out of order due to parallel execution,
                        # each task's index is captured in its closure when track_progress is created.
                        if (
                            task_index < len(pair_info)
                            and pair_info[task_index][1] is None
                        ):
                            await progress.update(1)  # Single node processed
                        else:
                            await progress.update(2)  # Pair processed

                    # Log batch completion every 10 tasks
                    completed_count += 1
                    if completed_count % 10 == 0 and overall_start_time:
                        if not (
                            progress
                            and progress.tracker
                            and progress.tracker.show_progress
                        ):
                            elapsed = time.time() - overall_start_time
                            mins, secs = divmod(int(elapsed), 60)
                            logger.info(
                                f"  Completed {completed_count}/{len(tasks)} pairs at height {current_height} [{mins}m {secs}s elapsed total]"
                            )

                    return result

                # Create tracked tasks
                tracked_tasks = [
                    track_progress(task, i) for i, task in enumerate(tasks)
                ]

                # Process all tasks concurrently (semaphore already controls parallelism)
                results = await asyncio.gather(*tracked_tasks)

                # Batch generate embeddings for all summaries at this level
                # This avoids individual API calls per node (e.g., 183 calls → 3 batch calls)
                summaries = [r["summary"] for r in results]

                start_time = time.time()
                embeddings = await self._get_embeddings_batch(summaries)

                # Track batch embedding call for telemetry
                if reporter:
                    node_embeddings = []
                    for result in results:
                        # Use the token count from summarization
                        token_count = result.get(
                            "token_count",
                            len(self.splitter.tokenizer.encode(result["summary"])),
                        )
                        node_embeddings.append((result["parent_id"], token_count))

                    reporter.record_embedding_call_v2(
                        node_embeddings=node_embeddings,
                        batch_size=len(summaries),
                        model=self.config.embedding_model,
                        start_time=start_time,
                    )

                # Update results with the generated embeddings
                for result, embedding in zip(results, embeddings):
                    result["node_data"]["embedding"] = embedding

                # Extract data for batch processing
                nodes_to_add = []
                parent_updates = []
                next_level_ids = []
                next_level_texts = []

                # Track preceding node for this level
                preceding_node_id = None

                for result in results:
                    # Add preceding neighbor ID to node data
                    result["node_data"]["preceding_neighbor_id"] = preceding_node_id

                    # Add node data for batch insertion
                    nodes_to_add.append(result["node_data"])

                    # Collect parent updates
                    for update in result["parent_updates"]:
                        if (
                            update is not None
                        ):  # Skip None updates (for nodes without right child)
                            parent_updates.append(update)

                    # Track IDs and texts for next level
                    next_level_ids.append(result["parent_id"])
                    next_level_texts.append(result["summary"])

                    # Update preceding ID for next iteration
                    preceding_node_id = result["parent_id"]

                # Batch store all nodes for this level
                if nodes_to_add:
                    self.store.add_nodes_batch(nodes_to_add)

                # Batch update all parent references
                if parent_updates:
                    self.store.update_parent_references_batch(parent_updates)

            current_level_ids = next_level_ids
            current_level_texts = next_level_texts

            # Track tree level completion
            if reporter and current_level_ids:
                try:
                    reporter.record_tree_height_complete(
                        current_height, len(current_level_ids)
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to record telemetry for tree level completion: {e}"
                    )

            current_height += 1

        # Return root node ID
        if current_level_ids:
            if overall_start_time:
                elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(elapsed), 60)
                if not (
                    progress and progress.tracker and progress.tracker.show_progress
                ):
                    logger.info(
                        f"Tree building complete. Root node at height {current_height - 1} with ID: {current_level_ids[0][:8]}... [{mins}m {secs}s elapsed total]"
                    )
            else:
                if not (
                    progress and progress.tracker and progress.tracker.show_progress
                ):
                    logger.info(
                        f"Tree building complete. Root node at height {current_height - 1} with ID: {current_level_ids[0][:8]}..."
                    )
        return current_level_ids[0] if current_level_ids else ""
