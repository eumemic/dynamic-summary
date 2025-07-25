"""Tree building and indexing functionality for RagZoom.

Note: Throughout this module, info-level logging is suppressed when show_progress=True
to prevent log messages from disrupting the progress bar display. This is why you'll see
`if not show_progress:` conditions before logger.info() calls.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Optional, cast

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN

from ragzoom.config import RagZoomConfig
from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker
from ragzoom.splitter import TextSplitter
from ragzoom.store import Store

logger = logging.getLogger(__name__)


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

    async def _summarize_text(
        self,
        left_text: str,
        right_text: str,
        target_tokens: int,
        prev_context: Optional[str] = None,
        next_context: Optional[str] = None,
    ) -> str:
        """Summarize text using LLM."""
        # Build prompt with adjacent context (trim to avoid token explosion)
        prompt_parts = []

        # Process adjacent context if needed
        trimmed_prev = None
        trimmed_next = None

        if prev_context and self.config.adjacent_context_tokens > 0:
            # Trim prev_context to adjacent_context_tokens
            prev_tokens = self.splitter.tokenizer.encode(prev_context)
            if len(prev_tokens) > self.config.adjacent_context_tokens:
                context_tokens = prev_tokens[-self.config.adjacent_context_tokens :]
                trimmed_prev = self.splitter.tokenizer.decode(context_tokens)
            else:
                trimmed_prev = prev_context

        if next_context and self.config.adjacent_context_tokens > 0:
            # Trim next_context to adjacent_context_tokens
            next_tokens = self.splitter.tokenizer.encode(next_context)
            if len(next_tokens) > self.config.adjacent_context_tokens:
                context_tokens = next_tokens[: self.config.adjacent_context_tokens]
                trimmed_next = self.splitter.tokenizer.decode(context_tokens)
            else:
                trimmed_next = next_context

        # Build the summarization prompt
        instruction = f"""You are an expert summarizer. You will be given a piece of content to summarize, embedded within the context in which it appears in a source document. You are to summarize only the content between the <SUMMARIZE_TEXT> tags in ≤{target_tokens} tokens, using the <PRECEDING_TEXT> and <FOLLOWING_TEXT> content as context (when provided - these may be omitted if there is no preceding/following context). The summary should be ≤{target_tokens} tokens in total. You should be able to substitute your summary where the <SUMMARIZE_TEXT> content is and it should work just as well within the context as the original text did. The <PRECEDING_TEXT> should flow smoothly into your summary and your summary should flow smoothly into the <FOLLOWING_TEXT>.

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

        prompt_parts.append(instruction)

        # Add preceding context if available
        if prev_context and self.config.adjacent_context_tokens > 0 and trimmed_prev:
            prompt_parts.append(
                f"\n<PRECEDING_TEXT>\n...{trimmed_prev.strip()}\n</PRECEDING_TEXT>"
            )

        # Add the content to summarize (concatenated)
        combined_text = f"{left_text} {right_text}".strip()
        prompt_parts.append(f"\n<SUMMARIZE_TEXT>\n{combined_text}\n</SUMMARIZE_TEXT>")

        # Add following context if available
        if next_context and self.config.adjacent_context_tokens > 0 and trimmed_next:
            prompt_parts.append(
                f"\n<FOLLOWING_TEXT>\n{trimmed_next.strip()}...\n</FOLLOWING_TEXT>"
            )

        full_prompt = "\n\n".join(prompt_parts)

        async with self.semaphore:
            try:
                response = await self.client.chat.completions.create(
                    model=self.config.summary_model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a precise summarizer who ONLY uses information explicitly provided in the input text. You NEVER add context or details from outside the given text.",
                        },
                        {"role": "user", "content": full_prompt},
                    ],
                    temperature=self.config.summary_temperature,
                    # No max_tokens limit - let LLM decide based on prompt instructions
                )
                content = response.choices[0].message.content
                summary = content.strip() if content else ""

            except Exception as e:
                logger.error(f"Error summarizing text: {e}")
                raise

        return summary

    async def _add_document_impl(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> str:
        """Add a document to the tree, creating leaf nodes."""
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
                leaf_ids.append(node_id)

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
                leaf_ids, chunks, document_id, async_progress, overall_start_time
            )

            # Final completion logging with total elapsed time
            if root_id:
                total_elapsed = time.time() - overall_start_time
                mins, secs = divmod(int(total_elapsed), 60)
                if not show_progress:
                    logger.info(
                        f"Document indexed successfully: {document_id} [{mins}m {secs}s total elapsed]"
                    )

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
    ) -> str:
        """Sync wrapper for add_document."""
        return asyncio.run(
            self.add_document_async(text, document_id, file_path, show_progress)
        )

    async def add_document_async(
        self,
        text: str,
        document_id: Optional[str] = None,
        file_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> str:
        """Async version of add_document - called by sync wrapper."""
        return await self._add_document_impl(
            text, document_id, file_path, show_progress
        )

    async def _process_node_pair(
        self,
        left_id: str,
        left_text: str,
        right_id: str,
        right_text: str,
        prev_context: Optional[str],
        next_context: Optional[str],
        document_id: Optional[str],
    ) -> tuple[str, str, list[float]]:
        """Process a single node pair - generate summary and embedding."""
        parent_id = self._generate_node_id()

        # Use consistent token budget for all heights
        # Target tokens for the summary (guidance for LLM, not hard limit)
        target_tokens = self.config.leaf_tokens

        # Generate summary (async)
        summary = await self._summarize_text(
            left_text, right_text, target_tokens, prev_context, next_context
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

        return parent_id, summary, embedding

    async def _build_tree_from_leaves(
        self,
        leaf_ids: list[str],
        leaf_texts: list[str],
        document_id: Optional[str] = None,
        progress: Optional[AsyncProgressWrapper] = None,
        overall_start_time: Optional[float] = None,
    ) -> str:
        """Build tree bottom-up from leaf nodes with concurrent processing."""
        current_level_ids = leaf_ids
        current_level_texts = leaf_texts

        # Calculate total tree height (distance from root to furthest leaf)
        # Note: This is used for progress tracking estimation

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
                next_context = None

                if i > 0:
                    prev_context, _ = self.splitter.get_adjacent_context(
                        current_level_texts, i - 1
                    )

                if i + 2 < len(current_level_texts):
                    _, next_context = self.splitter.get_adjacent_context(
                        current_level_texts, i + 1
                    )

                # Create async task

                task = self._process_node_pair(
                    left_id,
                    left_text,
                    right_id,
                    right_text,
                    prev_context,
                    next_context,
                    document_id,
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
                    if not odd_node_obj:
                        logger.error(f"Failed to retrieve odd node: {odd_node}")
                        raise ValueError("Odd node not found in store")

                    # For single-child parent, summary is essentially the child's text
                    # but may be slightly condensed to fit token budget
                    # odd_node_text is guaranteed to be non-None here due to the if condition
                    assert odd_node_text is not None
                    summary = await self._summarize_text(
                        odd_node_text,
                        "",  # No right child
                        self.config.leaf_tokens,
                        prev_context=None,
                        next_context=None,
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

            current_level_ids = next_level_ids
            current_level_texts = next_level_texts
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

    async def refresh_nodes_async(self, node_ids: list[str]) -> int:
        """Refresh dirty nodes by re-summarizing their content.

        Args:
            node_ids: List of node IDs to refresh

        Returns:
            Number of nodes successfully refreshed
        """
        refreshed_count = 0

        for node_id in node_ids:
            try:
                node = self.store.get_node(node_id)
                if not node or self.store.is_leaf_node(node_id):
                    # Skip leaf nodes - they don't have summaries
                    continue

                # Get children
                left_child, right_child = self.store.get_children(node_id)
                if not left_child or not right_child:
                    logger.warning(f"Node {node_id} missing children, skipping refresh")
                    continue

                # Re-summarize with fresh content
                summary = await self._summarize_text(
                    left_child.text,
                    right_child.text,
                    target_tokens=self.config.leaf_tokens,
                    prev_context="",
                    next_context="",
                )

                # Get new embedding
                embedding = await self._get_embedding(summary)

                # Update the node
                self.store.update_summary(node_id, summary, embedding)
                refreshed_count += 1

                logger.info(f"Refreshed node {node_id} with new summary")

            except Exception as e:
                logger.error(f"Error refreshing node {node_id}: {e}")
                continue

        return refreshed_count
