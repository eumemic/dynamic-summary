"""Tree building and indexing functionality for RagZoom."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, cast, overload

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN
from openai.types.chat import ChatCompletionMessageParam

from ragzoom.config import RagZoomConfig
from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker
from ragzoom.splitter import TextSplitter
from ragzoom.store import Store
from ragzoom.telemetry_collection import TelemetryCollector

logger = logging.getLogger(__name__)


class TreeBuilder:
    """Tree builder with concurrent processing."""

    def __init__(
        self,
        config: RagZoomConfig,
        store: Store,
        max_concurrent: int = 10,
    ):
        """Initialize tree builder.

        Args:
            config: RagZoom configuration
            store: Store instance for persistence
            max_concurrent: Maximum concurrent API requests
        """
        self.config = config
        self.store = store
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.splitter = TextSplitter(config)

    def _generate_node_id(self) -> str:
        """Generate unique node ID."""
        return str(uuid.uuid4())

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
                    dimensions=(
                        self.config.embedding_dimensions
                        if self.config.embedding_dimensions is not None
                        else NOT_GIVEN
                    ),
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
                    dimensions=(
                        self.config.embedding_dimensions
                        if self.config.embedding_dimensions is not None
                        else NOT_GIVEN
                    ),
                )
                return [item.embedding for item in response.data]
            except Exception as e:
                logger.error(f"Error getting batch embeddings: {e}")
                raise

    def _calculate_target_tokens(self, text: str) -> int:
        """Calculate target tokens as min of leaf_tokens or half the text size."""
        tokens = self.splitter.tokenizer.encode(text)
        half_size = len(tokens) // 2
        return min(self.config.leaf_tokens, half_size)

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
        api_kwargs = {
            "model": self.config.summary_model,
            "messages": messages,  # type: ignore
            "temperature": self.config.summary_temperature,
        }

        # Only add max_tokens if specified
        if target_tokens is not None:
            api_kwargs["max_tokens"] = int(target_tokens * 1.5)  # Safety margin

        response = await self.client.chat.completions.create(**api_kwargs)  # type: ignore

        content = response.choices[0].message.content
        if not content:
            logger.warning(f"{node_info}Empty response from LLM")
            raise ValueError("Empty response from LLM")

        summary = content.strip()
        if not summary:
            logger.warning(f"{node_info}Summary is empty after stripping whitespace")
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

    def _should_retry_summary(self, deviation_pct: float) -> bool:
        """Check if summary should be retried based on deviation from target.

        Args:
            deviation_pct: Percentage deviation from target tokens

        Returns:
            True if retry is needed, False otherwise
        """
        return deviation_pct > self.config.summary_deviation_threshold

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
    ) -> tuple[str, int, Any] | None:
        """Execute a single retry attempt.

        Args:
            messages: Conversation history
            target_tokens: Target token count
            node_info: Node identifier for logging
            summary: Current summary to append to conversation

        Returns:
            Tuple of (new_summary, token_count, response) or None if failed
        """
        try:
            # Calculate deviation for retry prompt
            current_tokens = len(self.splitter.tokenizer.encode(summary))
            deviation_pct = abs((current_tokens - target_tokens) / target_tokens)

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

        for retry_count in range(1, self.config.summary_max_retries + 1):
            # Check if we should retry
            current_tokens = len(self.splitter.tokenizer.encode(best_summary))
            deviation_pct = abs((current_tokens - target_tokens) / target_tokens)

            if not self._should_retry_summary(deviation_pct):
                return best_summary, actual_retries, best_attempt_index

            logger.debug(
                f"{node_info}Summary deviation: {current_tokens} tokens "
                f"(target: {target_tokens}, {deviation_pct:.1%} off). "
                f"Retry {retry_count}/{self.config.summary_max_retries}"
            )

            # Execute retry attempt
            retry_start = time.time()
            result = await self._execute_retry_attempt(
                messages, target_tokens, node_info, best_summary
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
    ) -> tuple[str, int]:
        """Summarize text using LLM.

        Returns:
            tuple: (summary, retry_count)
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

            # Don't record passthrough as a summary attempt - it pollutes metrics

            return combined_text, 0  # No retries needed

        # Build prompt with adjacent context (trim to avoid token explosion)
        prompt_parts: list[str] = []

        # Process adjacent context if needed
        trimmed_prev = None

        if prev_context and self.config.adjacent_context_tokens > 0:
            # Trim prev_context to adjacent_context_tokens
            prev_tokens = self.splitter.tokenizer.encode(prev_context)
            if len(prev_tokens) > self.config.adjacent_context_tokens:
                context_tokens = prev_tokens[-self.config.adjacent_context_tokens :]
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
        if prev_context and self.config.adjacent_context_tokens > 0 and trimmed_prev:
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
                except ValueError as e:
                    logger.warning(f"{node_info}{e}, using original text")
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

                if deviation_pct > self.config.summary_deviation_threshold:
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
                    reporter.mark_accepted_attempt(parent_id, accepted_attempt)

            except Exception as e:
                logger.error(f"Error summarizing text: {e}")
                raise

        return summary, retry_count

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
            lambda: validate_chunk_sizes(chunk_objects, self.config.leaf_tokens),
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
                    )
                leaf_ids.append(node_id)

                # Track chunk creation
                if reporter:
                    chunk_tokens = len(self.splitter.tokenizer.encode(chunk))
                    reporter.record_chunk_created(node_id, chunk_tokens)

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

            # Store all leaf nodes
            for i, (data, embedding) in enumerate(zip(chunk_data, all_embeddings)):
                self.store.add_node(
                    node_id=cast(str, data["id"]),
                    text=cast(str, data["text"]),
                    embedding=embedding,
                    span_start=cast(int, data["span_start"]),
                    span_end=cast(int, data["span_end"]),
                    document_id=document_id,
                )

            # Add document record
            if not existing_doc:
                self.store.add_document(
                    document_id, file_path, content_hash, len(chunks)
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
        right_id: str,
        right_text: str,
        prev_context: str | None,
        document_id: str | None,
        reporter: TelemetryCollector | None = None,
    ) -> tuple[str, str, list[float]]:
        """Process a single node pair - generate summary and embedding."""
        parent_id = self._generate_node_id()

        # Get node data for span information
        left_node = self.store.get_node(left_id)
        right_node = self.store.get_node(right_id)

        if not left_node or not right_node:
            logger.error(
                f"Failed to retrieve child nodes: left={left_id}, right={right_id}"
            )
            raise ValueError("Child nodes not found in store")

        # Track parent node creation
        if reporter:
            reporter.track_node_created(
                node_id=parent_id,
                height=reporter._current_height + 1,  # Parent is one level higher
            )

        # Use consistent token budget for all heights
        # Target tokens for the summary (guidance for LLM, not hard limit)
        target_tokens = self.config.leaf_tokens

        # Generate summary (async) with retry mechanism support
        summary, retry_count = await self._summarize_text(
            left_text,
            right_text,
            target_tokens,
            prev_context,
            parent_id,
            reporter=reporter,
        )

        # Get embedding for the summary
        start_time = time.time()
        embedding = await self._get_embedding(summary)

        # Track embedding for parent node
        if reporter and parent_id:
            summary_tokens = len(self.splitter.tokenizer.encode(summary))
            reporter.record_embedding_call_v2(
                node_embeddings=[(parent_id, summary_tokens)],
                batch_size=1,
                model=self.config.embedding_model,
                start_time=start_time,
            )

        # Store the node data
        self.store.add_node(
            node_id=parent_id,
            text=summary,
            embedding=embedding,
            span_start=left_node.span_start,
            span_end=right_node.span_end,
            left_child_id=left_id,
            right_child_id=right_id,
            document_id=document_id,
        )

        # Update children's parent references
        self._update_parent_reference(left_id, parent_id)
        self._update_parent_reference(right_id, parent_id)

        # Early validation: Check tree structure immediately after creating parent
        from ragzoom.validate import validate

        def check_parent_structure() -> str | None:
            # Check span validity
            if left_node.span_start >= right_node.span_end:
                return f"Invalid parent span: left child starts at {left_node.span_start}, right child ends at {right_node.span_end}"

            # Skip gap check in early validation - we'll check it properly in final validation
            # where we have access to the original text to verify if gaps are just whitespace

            return None

        validate(check_parent_structure, f"tree structure for parent {parent_id}")

        return parent_id, summary, embedding

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
            reporter.record_tree_height_complete(0, len(leaf_ids))

        current_height = 1  # Track height for logging (leaves are at height 0)
        while len(current_level_ids) > 1:
            next_level_ids = []
            next_level_texts = []
            # Note: current_height will be incremented after processing this height

            # Process pairs concurrently
            tasks = []
            pair_info = []

            # Handle odd number of nodes - keep last one unpaired
            nodes_to_pair = len(current_level_ids)
            if nodes_to_pair % 2 == 1:
                nodes_to_pair -= 1  # We'll handle the last node separately

            # Prepare all pairs
            for i in range(0, nodes_to_pair, 2):
                left_id = current_level_ids[i]
                left_text = current_level_texts[i]
                right_id = current_level_ids[i + 1]
                right_text = current_level_texts[i + 1]

                # Get adjacent context
                prev_context = None

                if i > 0:
                    prev_context, _ = self.splitter.get_adjacent_context(
                        current_level_texts, i - 1
                    )

                # Create async task

                task = self._process_node_pair(
                    left_id,
                    left_text,
                    right_id,
                    right_text,
                    prev_context,
                    document_id,
                    reporter,
                )
                tasks.append(task)
                pair_info.append((i, i + 1))

            # If there's an odd node at the end, create a parent with only left child
            # This ensures all leaves remain at the same depth
            odd_node = None
            odd_node_text = None
            if nodes_to_pair < len(current_level_ids):
                odd_node = current_level_ids[-1]
                odd_node_text = current_level_texts[-1]

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
                    if progress:
                        await progress.update(2)

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

                # Process tasks in smaller groups for better progress feedback
                results = []
                group_size = 20  # Process 20 at a time

                for i in range(0, len(tracked_tasks), group_size):
                    group = tracked_tasks[i : i + group_size]
                    group_results = await asyncio.gather(*group)
                    results.extend(group_results)

                # Add the parent nodes to next height
                for parent_id, summary, _ in results:
                    next_level_ids.append(parent_id)
                    next_level_texts.append(summary)

                # Handle odd node by creating a single-child parent
                if odd_node:
                    # Create a parent node with only a left child
                    parent_id = self._generate_node_id()

                    # Get the odd node for its span information
                    odd_node_obj = self.store.get_node(odd_node)

                    # Track parent node creation for telemetry
                    if reporter and odd_node_obj:
                        reporter.track_node_created(
                            node_id=parent_id,
                            height=current_height,  # Use current height
                        )
                    if not odd_node_obj:
                        logger.error(f"Failed to retrieve odd node: {odd_node}")
                        raise ValueError("Odd node not found in store")

                    # For single-child parent, summary is essentially the child's text
                    # but may be slightly condensed to fit token budget
                    # odd_node_text is guaranteed to be non-None here due to the if condition
                    assert odd_node_text is not None
                    summary, retry_count = await self._summarize_text(
                        odd_node_text,
                        "",  # No right child
                        self.config.leaf_tokens,
                        prev_context=None,
                        parent_id=parent_id,
                        reporter=reporter,
                    )

                    # Get embedding for the summary
                    start_time = time.time()
                    embedding = await self._get_embedding(summary)

                    # Track embedding for parent node
                    if reporter:
                        summary_tokens = len(self.splitter.tokenizer.encode(summary))
                        reporter.record_embedding_call_v2(
                            node_embeddings=[(parent_id, summary_tokens)],
                            batch_size=1,
                            model=self.config.embedding_model,
                            start_time=start_time,
                        )

                    # Store the single-child parent node
                    self.store.add_node(
                        node_id=parent_id,
                        text=summary,
                        embedding=embedding,
                        span_start=odd_node_obj.span_start,
                        span_end=odd_node_obj.span_end,
                        left_child_id=odd_node,
                        right_child_id=None,  # No right child
                        document_id=document_id,
                    )

                    # Update the odd node's parent reference
                    self._update_parent_reference(odd_node, parent_id)

                    # Update progress if tracking
                    if progress:
                        await progress.update(1)  # Count the odd node as processed

                    # Add the new parent to the next level
                    next_level_ids.append(parent_id)
                    next_level_texts.append(summary)

            current_level_ids = next_level_ids
            current_level_texts = next_level_texts

            # Track tree level completion
            if reporter and current_level_ids:
                reporter.record_tree_height_complete(
                    current_height, len(current_level_ids)
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
