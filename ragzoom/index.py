"""Tree building and indexing functionality for RagZoom."""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional, Union, cast, overload

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN

from ragzoom.config import RagZoomConfig
from ragzoom.metrics import IndexingMetrics, IndexingMetricsReporter
from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker
from ragzoom.splitter import TextSplitter
from ragzoom.store import Store

logger = logging.getLogger(__name__)

# Optional tqdm import for progress bar text output
try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None


class TreeBuilder:
    """Tree builder with concurrent processing."""

    def __init__(self, config: RagZoomConfig, store: Store, max_concurrent: int = 10):
        """Initialize tree builder.

        Args:
            config: RagZoom configuration
            store: Storage backend
            max_concurrent: Maximum concurrent API requests (default: 10)
        """
        self.config = config
        self.store = store
        self.splitter = TextSplitter(config)
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.semaphore = asyncio.Semaphore(max_concurrent)

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
        """Get embeddings for multiple texts in a single request."""
        async with self.semaphore:
            try:
                # OpenAI batch endpoint supports up to 2048 texts
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

    async def _retry_summary_correction(
        self,
        initial_summary: str,
        target_tokens: int,
        parent_id: Optional[str] = None,
        debug: bool = False,
    ) -> tuple[str, int]:
        """Retry summary correction to get closer to target token count.

        Returns:
            tuple: (best_summary, actual_retry_count)
        """
        summary = initial_summary
        summary_tokens = self.splitter.tokenizer.encode(summary)
        best_summary = summary
        best_token_count = len(summary_tokens)
        best_distance_from_target = abs(best_token_count - target_tokens)

        node_info = f"[{parent_id}] " if parent_id else ""
        actual_retries = 0

        for retry_count in range(1, self.config.summary_max_retries + 1):
            current_tokens = len(summary_tokens)
            deviation_pct = abs((current_tokens - target_tokens) / target_tokens)

            # Stop if within threshold
            if deviation_pct <= self.config.summary_deviation_threshold:
                return best_summary, actual_retries

            is_over_target = current_tokens > target_tokens

            if is_over_target:
                # Progressive reduction for over-target
                if retry_count <= len(self.config.summary_reduction_factors):
                    adjusted_target = int(
                        target_tokens
                        * self.config.summary_reduction_factors[retry_count - 1]
                    )
                else:
                    adjusted_target = int(target_tokens * 0.8)  # Fallback

                deviation = current_tokens - adjusted_target
                deviation_pct_display = (deviation / adjusted_target) * 100
                change_pct = (deviation / current_tokens) * 100

                actual_deviation_pct = (
                    (current_tokens - target_tokens) / target_tokens
                ) * 100
                if debug:
                    logger.info(
                        f"{node_info}Summary over target: {current_tokens} tokens "
                        f"(actual target: {target_tokens}, {actual_deviation_pct:.0f}% over). "
                        f"Retry {retry_count}/{self.config.summary_max_retries} - "
                        f"Asking LLM to target {adjusted_target} tokens."
                    )

                # Build reduction prompt
                if deviation_pct_display <= 50:
                    guidance = "Trim less essential details, descriptive passages, and minor events."
                elif deviation_pct_display <= 100:
                    guidance = "Focus only on major plot points and key character actions. Remove all minor details."
                else:
                    guidance = "Provide only the most critical events. Use extremely concise language."

                prompt = f"""Please revise this summary to be shorter.

Current summary ({current_tokens} tokens):
{summary}

Target: {adjusted_target} tokens (you are {deviation_pct_display:.0f}% over target)
To fix this, remove approximately {change_pct:.0f}% of the content.

{guidance}

Provide ONLY the shortened summary, with no preamble or explanation."""
            else:
                # Under target - request expansion
                deficit = target_tokens - current_tokens
                deficit_pct = (deficit / target_tokens) * 100
                expansion_pct = (deficit / current_tokens) * 100

                if debug:
                    logger.info(
                        f"{node_info}Summary under target: {current_tokens} tokens "
                        f"(target: {target_tokens}, {deficit_pct:.0f}% under). "
                        f"Retry {retry_count}/{self.config.summary_max_retries} - "
                        f"Can add {deficit} more tokens."
                    )

                # Build expansion prompt
                if deficit_pct <= 30:
                    guidance = (
                        "Add more specific details, exact names, numbers, and dates."
                    )
                elif deficit_pct <= 50:
                    guidance = "Include additional context, supporting details, and important descriptive elements."
                else:
                    guidance = "Significantly expand by including much more detail. Add complete sequences of events, full character actions, and comprehensive descriptions."

                prompt = f"""Please expand this summary with more detail.

Current summary ({current_tokens} tokens):
{summary}

Target: {target_tokens} tokens (you can add {deficit} more tokens)
To fix this, expand by approximately {expansion_pct:.0f}%.

{guidance}

Provide ONLY the expanded summary, with no preamble or explanation."""

            # Make retry request
            try:
                retry_response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=[
                        {
                            "role": "system",
                            "content": f"You are a precise editor who adjusts text length to meet specific token targets. Target: approximately {target_tokens} tokens.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.config.summary_temperature,
                    max_tokens=int(target_tokens * 1.5),  # Safety margin
                )

                content = retry_response.choices[0].message.content
                if not content:
                    logger.warning(f"{node_info}Empty response from LLM during retry")
                    continue

                summary = content.strip()
                actual_retries = retry_count  # Track that we performed this retry

                # Check new token count
                new_tokens = self.splitter.tokenizer.encode(summary)
                new_token_count = len(new_tokens)
                new_distance = abs(new_token_count - target_tokens)

                # Update best if improved
                if (
                    new_token_count <= target_tokens
                    and new_distance < best_distance_from_target
                ) or (
                    best_token_count > target_tokens
                    and new_token_count < best_token_count
                ):
                    best_summary = summary
                    best_token_count = new_token_count
                    best_distance_from_target = new_distance

                    if debug:
                        logger.info(
                            f"{node_info}Better result: {current_tokens} -> {new_token_count} tokens "
                            f"(target: {target_tokens}, distance: {new_distance})"
                        )
                else:
                    if debug:
                        logger.info(
                            f"{node_info}No improvement: {current_tokens} -> {new_token_count} tokens "
                            f"(best so far: {best_token_count} tokens)"
                        )

                # Use best for next iteration
                summary = best_summary
                summary_tokens = self.splitter.tokenizer.encode(best_summary)

            except Exception as e:
                logger.error(f"{node_info}Error during retry {retry_count}: {e}")
                break

        return best_summary, actual_retries

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        prev_context: Optional[str] = None,
        parent_id: Optional[str] = None,
        debug: bool = False,
        reporter: Optional[IndexingMetricsReporter] = None,
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
            logger.info(
                f"{node_info}Combined text already under target: {current_token_count} tokens "
                f"(target: {target_tokens}). Skipping summarization."
            )
            return combined_text, 0  # No retries needed
        # Build prompt with adjacent context (trim to avoid token explosion)
        prompt_parts = []

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

        # Calculate compression stats for the prompt
        tokens_to_remove = current_token_count - target_tokens
        compression_ratio = (tokens_to_remove / current_token_count) * 100

        # Build the summarization prompt
        instruction = f"""You are an expert summarizer. You will be given a piece of content to summarize, embedded within the context in which it appears in a source document. You are to summarize only the content between the <SUMMARIZE_TEXT> tags, using the <PRECEDING_TEXT> content as context (when provided - this may be omitted if there is no preceding context).

TOKEN REQUIREMENTS:
- The content to summarize is {current_token_count} tokens
- Your target is {target_tokens} tokens
- This means you need to compress {current_token_count} tokens into {target_tokens} tokens
- That's a {compression_ratio:.0f}% compression (remove {tokens_to_remove} tokens)

CRITICAL REQUIREMENTS:
- Summarize ONLY the content between the <SUMMARIZE_TEXT> and </SUMMARIZE_TEXT> tags
- Your summary should be approximately {target_tokens} tokens (aim for 90-100% of this target)
- The summary MUST NOT exceed {target_tokens} tokens. This is a HARD LIMIT.
- IMPORTANT: Use as close to {target_tokens} tokens as possible. Do not be overly brief.
- Include as much detail and information as will fit within the token budget
- Make the summary as information-dense as possible, preserving names, numbers, and specific details
- The summary should attempt to cover the full scope of the content from start to end, but abstract over details as necessary to stay under the token limit
- Focus on key events, facts, and themes ONLY from the provided text
- Do NOT include any concrete information from BEFORE the <SUMMARIZE_TEXT> tag or AFTER the </SUMMARIZE_TEXT> tag in the summary
- Do NOT complete a sentence that is cut off with information from outside the <SUMMARIZE_TEXT> block
- Do NOT infer or imagine details not present in the text
- If the text references something without explaining it, do NOT try to explain it
- IMPORTANT: Match voice and tense of the text you are summarizing! If you are summarizing a text written in the past tense, the summary MUST be in past tense
- Try to match the tone and style of the text, as if the summary were written by the same author
- Respond ONLY with your best attempt at a summary, do not break the fourth wall, say you can't summarize it, etc.

Here's the content to summarize:"""

        prompt_parts.append(instruction)

        # Add preceding context if available
        if prev_context and self.config.adjacent_context_tokens > 0 and trimmed_prev:
            prompt_parts.append(
                f"\n<PRECEDING_TEXT>\n...{trimmed_prev.strip()}\n</PRECEDING_TEXT>"
            )

        # Add the content to summarize (concatenated)
        combined_text = f"{left_text} {right_text}".strip()
        prompt_parts.append(f"\n<SUMMARIZE_TEXT>\n{combined_text}\n</SUMMARIZE_TEXT>")

        full_prompt = "\n\n".join(prompt_parts)

        async with self.semaphore:
            try:
                # Build initial messages for conversation
                messages: list[dict[str, Any]] = [
                    {
                        "role": "system",
                        "content": f"You are a precise summarizer who creates detailed summaries of approximately {target_tokens} tokens. You ONLY use information explicitly provided in the input text. You NEVER add context or details from outside the given text. Aim to use 90-100% of the available token budget.",
                    },
                    {"role": "user", "content": full_prompt},
                ]

                response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=messages,  # type: ignore
                    temperature=self.config.summary_temperature,
                    # No max_tokens limit - let LLM decide based on prompt instructions
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("Empty response from LLM")

                summary = content.strip()
                if not summary:
                    raise ValueError("Summary is empty after stripping whitespace")

                # Check if summary needs correction
                summary_tokens = self.splitter.tokenizer.encode(summary)
                current_tokens = len(summary_tokens)
                deviation_pct = abs((current_tokens - target_tokens) / target_tokens)

                node_info = f"[{parent_id}] " if parent_id else ""

                # Use retry correction if deviation exceeds threshold
                retry_count = 0
                if deviation_pct > self.config.summary_deviation_threshold:
                    summary, retry_count = await self._retry_summary_correction(
                        summary, target_tokens, parent_id, debug
                    )
                    # Re-calculate final tokens for logging
                    final_tokens = len(self.splitter.tokenizer.encode(summary))
                    utilization_pct = (final_tokens / target_tokens) * 100
                    if debug:
                        logger.info(
                            f"{node_info}Summary corrected: {final_tokens} tokens "
                            f"(target: {target_tokens}, {utilization_pct:.0f}% utilization)"
                        )
                else:
                    # Log initial result
                    utilization_pct = (current_tokens / target_tokens) * 100
                    if debug:
                        logger.info(
                            f"{node_info}Summary complete: {current_tokens} tokens "
                            f"(target: {target_tokens}, {utilization_pct:.0f}% utilization)"
                        )

                # Track summary result
                if reporter and hasattr(response, "usage") and response.usage:
                    summary_tokens_list = self.splitter.tokenizer.encode(summary)
                    summary_token_count = len(summary_tokens_list)
                    reporter.record_summary_result(
                        target_tokens=self.config.leaf_tokens,  # Always use configured target for metrics
                        actual_tokens=summary_token_count,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                    )

            except Exception as e:
                logger.error(f"Error summarizing text: {e}")
                raise

        return summary, retry_count

    @overload
    async def _add_document_impl(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = True,
        debug: bool = False,
        reporter: None = None,
    ) -> str: ...

    @overload
    async def _add_document_impl(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = True,
        debug: bool = False,
        reporter: IndexingMetricsReporter = ...,
    ) -> tuple[str, IndexingMetrics]: ...

    async def _add_document_impl(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = True,
        debug: bool = False,
        reporter: Optional[IndexingMetricsReporter] = None,
    ) -> Union[str, tuple[str, IndexingMetrics]]:
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

        if debug:
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

        # Create progress tracker right before we start actual work
        progress = (
            GlobalProgressTracker(len(chunks), show_progress) if show_progress else None
        )

        # Create async wrapper for progress
        async_progress = AsyncProgressWrapper(progress) if progress else None

        # Track overall start time for cumulative elapsed time
        overall_start_time = time.time()

        try:
            # Create leaf nodes with batch embeddings
            if len(chunks) > 100 and debug:
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
                if debug:
                    elapsed = time.time() - overall_start_time
                    mins, secs = divmod(int(elapsed), 60)
                    logger.info(
                        f"Processing embedding batch: chunks {i+1}-{batch_end} of {len(chunks)} [{mins}m {secs}s elapsed]"
                    )

                # Track embedding call
                if reporter:
                    token_counts = [
                        len(self.splitter.tokenizer.encode(text))
                        for text in batch_texts
                    ]
                    reporter.record_embedding_call(len(batch_texts), token_counts)

                batch_embeddings = await self._get_embeddings_batch(batch_texts)
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
                debug,
                reporter,
            )

            # Final completion logging with total elapsed time
            if root_id:
                total_elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(total_elapsed), 60)
                logger.info(
                    f"Document indexed successfully: {document_id} [{mins}m {secs}s total elapsed]"
                )

            # Finalize metrics if reporter was used
            if reporter:
                metrics = reporter.finalize()
                return document_id, metrics

            return document_id
        finally:
            # Always close progress
            if progress:
                progress.close()

    def add_document(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = True,
        debug: bool = False,
    ) -> str:
        """Sync wrapper for add_document."""
        return asyncio.run(
            self.add_document_async(text, document_id, file_path, show_progress, debug)
        )

    def add_document_with_metrics(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = False,
        debug: bool = False,
    ) -> tuple[str, IndexingMetrics]:
        """Add document and return metrics. Used for benchmarking.

        This is a convenience method that creates an IndexingMetricsReporter internally
        and returns the collected metrics. For production use, add_document() is preferred
        as it doesn't have the overhead of metrics collection.

        The dual-method pattern ensures:
        - Normal indexing (add_document) has zero metrics overhead
        - Benchmarking gets detailed metrics without modifying core logic
        - Internal implementation (_add_document_impl) remains flexible
        """
        # Create reporter internally with config for pricing
        source_tokens = len(self.splitter.tokenizer.encode(text))
        reporter = IndexingMetricsReporter(
            document_id or "benchmark", source_tokens, self.config
        )

        # Run indexing with reporter - will return (doc_id, metrics)
        result = asyncio.run(
            self._add_document_impl(
                text, document_id, file_path, show_progress, debug, reporter
            )
        )

        # Extract tuple returned when reporter is provided
        # Type checker knows result is a tuple because we passed a reporter
        doc_id, metrics = result
        return doc_id, metrics

    async def add_document_async(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = True,
        debug: bool = False,
    ) -> str:
        """Async version of add_document - called by sync wrapper."""
        result = await self._add_document_impl(
            text, document_id, file_path, show_progress, debug, reporter=None
        )
        # Type checker knows result is a string when no reporter is provided
        return cast(str, result)

    async def _process_node_pair(
        self,
        left_id: str,
        left_text: str,
        right_id: str,
        right_text: str,
        prev_context: Optional[str],
        document_id: Optional[str],
        debug: bool = False,
        reporter: Optional[IndexingMetricsReporter] = None,
    ) -> tuple[str, str, list[float], int]:
        """Process a single node pair - generate summary and embedding.

        Returns:
            tuple: (parent_id, summary, embedding, retry_count)
        """
        parent_id = self._generate_node_id()

        # Calculate target tokens
        combined_text = f"{left_text} {right_text}".strip()
        target_tokens = self._calculate_target_tokens(combined_text)

        # Generate summary (async)
        summary, retry_count = await self._summarize_text(
            left_text,
            right_text,
            target_tokens,
            prev_context,
            parent_id,
            debug,
            reporter,
        )

        # Validate summary faithfulness if validation is enabled
        # Note: For internal nodes, left_text and right_text are summaries from children,
        # not original text. The validation checks that the parent summary only contains
        # information from these child summaries.
        # from ragzoom.validate import validate_summary_faithfulness
        #
        # validation_error = await validate_summary_faithfulness(
        #     summary, left_text, right_text, self.client
        # )
        # if validation_error:
        #     # Create a comprehensive error message
        #     error_msg = f"\n{'='*80}\nSUMMARY VALIDATION FAILED\n{'='*80}\n"
        #     error_msg += f"Parent of: {left_id} (left), {right_id} (right)\n"
        #     error_msg += f"Reason: {validation_error}\n"
        #     error_msg += f"{'-'*80}\n"
        #     error_msg += f"Generated summary:\n{summary}\n"
        #     error_msg += f"{'-'*80}\n"
        #     error_msg += f"Left child:\n{left_text}\n"
        #     error_msg += f"{'-'*80}\n"
        #     error_msg += f"Right child:\n{right_text}\n"
        #     error_msg += f"{'='*80}"
        #
        #     logger.error(error_msg)
        #     raise ValueError(validation_error)

        # Get embedding for the summary
        embedding = await self._get_embedding(summary)

        # Store the node data
        left_node = self.store.get_node(left_id)
        right_node = self.store.get_node(right_id)

        if not left_node or not right_node:
            logger.error(
                f"Failed to retrieve child nodes: left={left_id}, right={right_id}"
            )
            raise ValueError("Child nodes not found in store")

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

        def check_parent_structure() -> Optional[str]:
            # Check span validity
            if left_node.span_start >= right_node.span_end:
                return f"Invalid parent span: left child starts at {left_node.span_start}, right child ends at {right_node.span_end}"

            # Skip gap check in early validation - we'll check it properly in final validation
            # where we have access to the original text to verify if gaps are just whitespace

            return None

        validate(check_parent_structure, f"tree structure for parent {parent_id}")

        return parent_id, summary, embedding, retry_count

    async def _build_tree_from_leaves(
        self,
        leaf_ids: list[str],
        leaf_texts: list[str],
        document_id: Optional[str] = None,
        progress: Optional[AsyncProgressWrapper] = None,
        overall_start_time: Optional[float] = None,
        debug: bool = False,
        reporter: Optional[IndexingMetricsReporter] = None,
    ) -> str:
        """Build tree bottom-up from leaf nodes with concurrent processing."""
        current_level_ids = leaf_ids
        current_level_texts = leaf_texts

        # Calculate total tree height (distance from root to furthest leaf)
        # Note: This is used for progress tracking estimation

        # Track token usage statistics by height (keep for debug logging)
        token_stats: dict[int, list[int]] = {}

        # Record leaf node token counts (height 0)
        token_stats[0] = []
        for text in leaf_texts:
            tokens = self.splitter.tokenizer.encode(text)
            token_stats[0].append(len(tokens))

        # Track leaf level
        if reporter:
            reporter.record_tree_level_complete(0, len(leaf_ids))

        current_height = 1  # Track height for logging (leaves are at height 0)
        while len(current_level_ids) > 1:
            next_level_ids = []
            next_level_texts = []
            # Note: current_height will be incremented after processing this height

            # Initialize token stats for this height
            token_stats[current_height] = []

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
                    debug,
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
                # Show tree building progress inline with progress bar
                if progress and progress.tracker and progress.tracker.pbar and HAS_TQDM:
                    if overall_start_time:
                        elapsed = time.time() - overall_start_time
                        mins, secs = divmod(int(elapsed), 60)
                        tqdm.write(
                            f"Building tree height {current_height}: processing {len(tasks)} node pairs [{mins}m {secs}s elapsed]"
                        )
                    else:
                        tqdm.write(
                            f"Building tree height {current_height}: processing {len(tasks)} node pairs"
                        )
                elif debug:
                    # Only log to stderr if no progress bar and debug is enabled
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

                    # Log batch completion only in debug mode and at larger intervals
                    completed_count += 1
                    if completed_count % 20 == 0 and overall_start_time and debug:
                        elapsed = time.time() - overall_start_time
                        mins, secs = divmod(int(elapsed), 60)
                        if (
                            progress
                            and progress.tracker
                            and progress.tracker.pbar
                            and HAS_TQDM
                        ):
                            tqdm.write(
                                f"  Completed {completed_count}/{len(tasks)} pairs at height {current_height} [{mins}m {secs}s elapsed total]"
                            )
                        else:
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
                for parent_id, summary, _, retry_count in results:
                    next_level_ids.append(parent_id)
                    next_level_texts.append(summary)

                    # Track retry in reporter if available
                    if reporter:
                        reporter.record_summary_retry(current_height, retry_count)

                # Handle odd node by creating a single-child parent
                if odd_node:
                    # Create a parent node with only a left child
                    parent_id = self._generate_node_id()

                    # Get the odd node for its span information
                    odd_node_obj = self.store.get_node(odd_node)
                    if not odd_node_obj:
                        logger.error(f"Failed to retrieve odd node: {odd_node}")
                        raise ValueError("Odd node not found in store")

                    # For single-child parent, summary is essentially the child's text
                    # but may be slightly condensed to fit token budget
                    # odd_node_text is guaranteed to be non-None here due to the if condition
                    assert odd_node_text is not None

                    # Calculate target tokens for single-child case
                    target_tokens = self._calculate_target_tokens(odd_node_text)

                    summary, retry_count = await self._summarize_text(
                        odd_node_text,
                        "",  # No right child
                        target_tokens,
                        prev_context=None,
                        parent_id=parent_id,
                        debug=debug,
                        reporter=reporter,
                    )

                    # Get embedding for the summary
                    embedding = await self._get_embedding(summary)

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

                    # Track retry in reporter if available
                    if reporter:
                        reporter.record_summary_retry(current_height, retry_count)

                # Track token counts for all nodes at this height
                for text in next_level_texts:
                    tokens = self.splitter.tokenizer.encode(text)
                    token_stats[current_height].append(len(tokens))

            current_level_ids = next_level_ids
            current_level_texts = next_level_texts

            # Track tree level completion
            if reporter and current_level_ids:
                reporter.record_tree_level_complete(
                    current_height, len(current_level_ids)
                )

            current_height += 1

        # Return root node ID
        if current_level_ids:
            if debug:
                if overall_start_time:
                    elapsed = time.time() - overall_start_time
                    mins, secs = divmod(int(elapsed), 60)
                    logger.info(
                        f"Tree building complete. Root node at height {current_height - 1} with ID: {current_level_ids[0][:8]}... [{mins}m {secs}s elapsed total]"
                    )
                else:
                    logger.info(
                        f"Tree building complete. Root node at height {current_height - 1} with ID: {current_level_ids[0][:8]}..."
                    )

            # Log token usage and retry statistics if debug is enabled
            if debug:
                logger.info("\nToken usage statistics by tree height:")
                for height in sorted(token_stats.keys()):
                    counts = token_stats[height]
                    if counts:
                        avg_tokens = sum(counts) / len(counts)
                        min_tokens = min(counts)
                        max_tokens = max(counts)
                        logger.info(
                            f"  Height {height}: avg {avg_tokens:.0f} tokens, "
                            f"min {min_tokens}, max {max_tokens} ({len(counts)} nodes)"
                        )

        return current_level_ids[0] if current_level_ids else ""

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
