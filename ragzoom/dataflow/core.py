"""Core dataflow implementation for parallel tree indexing using TreeNode directly.

This module implements the dataflow pattern where nodes are processed
as soon as their dependencies are ready, enabling maximum parallelism.
"""

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from ragzoom.dataflow.domain import DomainNode as TreeNode
from ragzoom.progress import AsyncProgressWrapper
from ragzoom.services.llm_service import LLMService
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


class ProcessingStrategy(Enum):
    """Strategy for prioritizing nodes in the summary processing queue.

    Both strategies use dataflow parallelism - nodes process as soon as their
    dependencies are ready. The strategy only affects the ordering when multiple
    nodes are ready simultaneously.
    """

    BottomToTop = "bottom_to_top"  # Level-first, then left-to-right within level
    LeftToRight = "left_to_right"  # Position-first regardless of level


class BatchAwareQueue:
    """Queue that coordinates batching using condition variables.

    Workers sleep until enough items are available for a batch. The key insight
    is that we must flush the penultimate batch as soon as both root children
    complete (depth-1 nodes), not when the root arrives, to minimize latency.
    """

    def __init__(self, batch_size: int):
        """Initialize the batch-aware queue.

        Args:
            batch_size: Target batch size for processing
        """
        self.queue: asyncio.Queue[TreeNode] = asyncio.Queue()
        self.batch_size = batch_size
        self.condition = asyncio.Condition()
        self.root_children_complete = False
        self.root_queued = False
        self.depth_1_count = 0

    async def put(self, item: TreeNode) -> None:
        """Add item to queue and notify waiting workers.

        Args:
            item: TreeNode to add to the queue
        """
        async with self.condition:
            await self.queue.put(item)

            depth = item.get_depth()
            if depth == 0:
                # Root has arrived
                self.root_queued = True
            elif depth == 1 and not self.root_children_complete:
                # This is a child of root - increment counter
                self.depth_1_count += 1
                if self.depth_1_count >= 2:
                    self.root_children_complete = True

            self.condition.notify_all()

    async def get_batch(
        self, shutdown: asyncio.Event | None = None
    ) -> list[TreeNode] | None:
        """Get a batch of items, sleeping until ready.

        Args:
            shutdown: Optional shutdown event to check

        Returns:
            List of TreeNodes for processing, or None if queue is closing
        """
        async with self.condition:
            while True:
                if shutdown and shutdown.is_set():
                    return None

                size = self.queue.qsize()

                # Process if:
                # 1. Full batch available
                # 2. Root children complete -> flush penultimate batch immediately
                # 3. Root queued -> process singleton root batch
                should_process = (
                    size >= self.batch_size
                    or (
                        self.root_children_complete
                        and not self.root_queued
                        and size > 0
                    )
                    or (self.root_queued and size > 0)
                )

                if should_process:
                    batch = []
                    batch_size = min(size, self.batch_size)
                    for _ in range(batch_size):
                        try:
                            item = self.queue.get_nowait()
                            batch.append(item)
                        except asyncio.QueueEmpty:
                            break

                    if batch:
                        return batch

                elif self.root_queued and size == 0:
                    # All work complete
                    return None

                # Sleep until notified (with timeout for shutdown check)
                try:
                    await asyncio.wait_for(self.condition.wait(), timeout=0.5)
                except asyncio.TimeoutError:
                    continue

    def task_done(self) -> None:
        """Mark a task as done for join() compatibility."""
        self.queue.task_done()

    async def join(self) -> None:
        """Wait for all tasks to be marked done."""
        await self.queue.join()


@dataclass
class SummaryJob:
    """A summary generation job with priority ordering."""

    node: TreeNode
    strategy: ProcessingStrategy = field(default=ProcessingStrategy.BottomToTop)

    def __lt__(self, other: "SummaryJob") -> bool:
        """Determine priority ordering between two summary jobs.

        The comparison logic depends on the processing strategy:
        - BottomToTop: Prioritize lower tree levels first (height 0 before height 1),
          then leftmost nodes within each level (by span_start).
          This ensures all nodes at one level complete before the next begins.
        - LeftToRight: Prioritize strictly by document position (span_start),
          regardless of tree level. This creates a left-to-right wave through
          the document.

        Returns:
            True if this job should be processed before the other job.
        """
        if self.strategy == ProcessingStrategy.BottomToTop:
            # Process lower levels first, then leftmost within level
            if self.node.height != other.node.height:
                return self.node.height < other.node.height
            return self.node.span_start < other.node.span_start
        else:  # LEFT_TO_RIGHT
            # Process leftmost first regardless of level
            return self.node.span_start < other.node.span_start


def _generate_node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())


def create_leaf_nodes(
    chunks: list[str], document_id: str, reporter: TelemetryCollector | None = None
) -> tuple[dict[str, TreeNode], list[TreeNode]]:
    """Create leaf nodes from document chunks.

    Args:
        chunks: List of text chunks from the document
        document_id: ID of the document being indexed

    Returns:
        Tuple of (lookup dict, list of leaf nodes)
    """
    if not chunks:
        raise ValueError("No chunks provided")

    lookup: dict[str, TreeNode] = {}
    leaves: list[TreeNode] = []

    # Track position in document
    current_pos = 0

    previous_leaf = None
    for i, chunk in enumerate(chunks):
        node_id = _generate_node_id()

        # Calculate token count for this chunk
        chunk_tokens = tokenizer.count_tokens(chunk)

        # Create TreeNode with actual text for leaves
        leaf = TreeNode(
            id=node_id,
            text=chunk,  # Leaves have actual text
            height=0,  # Leaves are at height 0
            span_start=current_pos,
            span_end=current_pos + len(chunk),
            path="",
            document_id=document_id,
            # No parent/children for leaves initially
            parent_id=None,
            left_child_id=None,
            right_child_id=None,
            # Neighbor relationships
            preceding_neighbor_id=previous_leaf.id if previous_leaf else None,
            following_neighbor_id=None,  # Will be set by next leaf
            # Leaves need embeddings but we'll generate them later
            embedding=None,
            token_count=chunk_tokens,  # Set actual token count
        )

        # Update previous leaf's following_neighbor_id
        if previous_leaf:
            previous_leaf.following_neighbor_id = leaf.id

        leaves.append(leaf)
        lookup[node_id] = leaf

        # Log node creation
        logger.debug(
            f"Created leaf {node_id[:8]} h=0 span=[{current_pos}:{leaf.span_end}] tokens={chunk_tokens}"
        )

        # Track node creation for telemetry
        if reporter:
            reporter.track_node_created(
                node_id=leaf.id,
                height=0,
                span=(leaf.span_start, leaf.span_end),
            )
            reporter.record_chunk_created(leaf.id, chunk_tokens)

        previous_leaf = leaf
        current_pos = leaf.span_end

    return lookup, leaves


def build_internal_nodes(
    lookup: dict[str, TreeNode],
    leaves: list[TreeNode],
    document_id: str,
    reporter: TelemetryCollector | None = None,
) -> None:
    """Build internal nodes from leaves bottom-up.

    Modifies lookup in-place, adding all internal nodes.

    Args:
        lookup: Dictionary of node_id -> TreeNode to modify
        leaves: List of leaf nodes to build tree from
        document_id: Document ID for all nodes
    """
    if not leaves:
        return

    # Special case: single leaf is also the root
    if len(leaves) == 1:
        return

    current_level = leaves
    current_height = 1

    while len(current_level) > 1:
        parents: list[TreeNode] = []
        previous_parent = None

        i = 0
        while i < len(current_level):
            left = current_level[i]
            right = current_level[i + 1] if i + 1 < len(current_level) else None

            parent_id = _generate_node_id()

            # Create TreeNode with empty text (to be filled by dataflow)
            parent = TreeNode(
                id=parent_id,
                text="",  # Empty string until filled by dataflow
                height=current_height,
                span_start=left.span_start,
                span_end=right.span_end if right else left.span_end,
                path="",
                document_id=document_id,
                # Children
                parent_id=None,  # Will be set when grandparent created
                left_child_id=left.id,
                right_child_id=right.id if right else None,
                # Neighbors
                preceding_neighbor_id=previous_parent.id if previous_parent else None,
                following_neighbor_id=None,  # Will be set by next parent
                # No embedding attached yet; captured later
                embedding=None,
                token_count=0,
            )

            # Update relationships
            left.parent_id = parent.id
            if right:
                right.parent_id = parent.id
            if previous_parent:
                previous_parent.following_neighbor_id = parent.id

            parents.append(parent)
            lookup[parent.id] = parent

            # Log internal node creation
            logger.debug(
                f"Created node {parent.id[:8]} h={current_height} span=[{parent.span_start}:{parent.span_end}] "
                f"children=({left.id[:8]},{right.id[:8] if right else 'None'})"
            )

            # Track internal node creation for telemetry
            if reporter:
                reporter.track_node_created(
                    node_id=parent.id,
                    height=parent.height,
                    span=(parent.span_start, parent.span_end),
                )

            previous_parent = parent

            # Move to next pair
            i += 2 if right else 1

        current_level = parents
        current_height += 1

    if current_level:
        _assign_depths(current_level[0].id, lookup)


def _assign_depths(root_id: str, lookup: dict[str, TreeNode]) -> None:
    """Propagate depth (distance from root) across the local tree."""

    queue: deque[tuple[str, int]] = deque([(root_id, 0)])
    visited: set[str] = set()

    while queue:
        node_id, depth = queue.popleft()
        node = lookup.get(node_id)
        if not node or node_id in visited:
            continue
        visited.add(node_id)
        node.depth = depth

        left_id = getattr(node, "left_child_id", None)
        if left_id:
            queue.append((left_id, depth + 1))

        right_id = getattr(node, "right_child_id", None)
        if right_id:
            queue.append((right_id, depth + 1))


def poke(
    node_id: str,
    lookup: dict[str, TreeNode],
    queue: asyncio.PriorityQueue[SummaryJob],
    strategy: ProcessingStrategy,
) -> None:
    """Check if node's dependencies are ready and queue if so.

    Args:
        node_id: ID of node to check
        lookup: Dictionary containing all nodes
        queue: Priority queue to add node to if ready
        strategy: Processing strategy for priority ordering
    """
    node = lookup.get(node_id)
    if not node:
        return

    # Check if all dependencies are ready
    ready = True

    # Check children have text (non-empty for processed nodes)
    if node.left_child_id:
        left_child = lookup.get(node.left_child_id)
        if not left_child or not left_child.text:
            ready = False
        # Also check if left child's preceding neighbor is ready (for context)
        elif left_child.preceding_neighbor_id:
            left_child_preceding = lookup.get(left_child.preceding_neighbor_id)
            if not left_child_preceding or not left_child_preceding.text:
                ready = False

    if ready and node.right_child_id:
        right_child = lookup.get(node.right_child_id)
        if not right_child or not right_child.text:
            ready = False

    # Queue if ready with the specified processing strategy
    if ready:
        queue.put_nowait(SummaryJob(node, strategy))


async def summary_worker(
    worker_id: int,
    lookup: dict[str, TreeNode],
    summary_queue: asyncio.PriorityQueue[SummaryJob],
    embedding_queue: BatchAwareQueue,
    llm_service: LLMService,
    shutdown: asyncio.Event,
    processing_strategy: ProcessingStrategy,
    target_tokens: int = 200,
    reporter: TelemetryCollector | None = None,
    progress: AsyncProgressWrapper | None = None,
) -> None:
    """Worker that processes summary generation.

    Args:
        worker_id: ID for logging
        lookup: Dictionary with all nodes
        summary_queue: Queue of nodes ready for summary
        embedding_queue: Queue to send completed summaries for embedding
        llm_service: Service for generating summaries
        shutdown: Event to signal shutdown
    """
    while not shutdown.is_set():
        try:
            # Get next node to process (with timeout for shutdown check)
            try:
                job = await asyncio.wait_for(summary_queue.get(), timeout=0.1)
                node = job.node
            except asyncio.TimeoutError:
                continue

            try:

                # Get child texts and token counts
                left_text = ""
                right_text = ""
                left_token_count = 0
                right_token_count = 0

                if node.left_child_id:
                    left_child = lookup[node.left_child_id]
                    left_text = left_child.text or ""
                    left_token_count = left_child.token_count
                if node.right_child_id:
                    right_child = lookup[node.right_child_id]
                    right_text = right_child.text or ""
                    right_token_count = right_child.token_count

                # Get preceding context from left child's preceding neighbor
                prev_context = None
                if node.left_child_id:
                    left_child = lookup[node.left_child_id]
                    if left_child.preceding_neighbor_id:
                        left_child_preceding = lookup.get(
                            left_child.preceding_neighbor_id
                        )
                        if left_child_preceding:
                            prev_context = left_child_preceding.text

                # Generate summary - telemetry is handled internally by _summarize_text
                summary, retry_count, tokens = await llm_service._summarize_text(
                    left_text,
                    right_text,
                    target_tokens,
                    parent_id=node.id,  # Pass node.id as parent_id for telemetry
                    reporter=reporter,  # Pass reporter so telemetry works
                    prev_context=prev_context,
                    left_token_count=left_token_count,
                    right_token_count=right_token_count,
                )

                # Update node
                node.text = summary
                node.token_count = tokens

                # Log summary completion
                logger.debug(
                    f"Summary {node.id[:8]} h={node.height} tokens={tokens} attempt={retry_count + 1}"
                )

                parent = lookup.get(node.parent_id) if node.parent_id else None
                if parent:
                    poke(parent.id, lookup, summary_queue, processing_strategy)

                    if parent.right_child_id == node.id and node.following_neighbor_id:
                        following_neighbor = lookup.get(node.following_neighbor_id)
                        if following_neighbor and following_neighbor.parent_id:
                            poke(
                                following_neighbor.parent_id,
                                lookup,
                                summary_queue,
                                processing_strategy,
                            )

                # Queue for embedding (will notify waiting workers)
                await embedding_queue.put(node)

                # Update progress synchronously
                if progress:
                    progress.update_sync(1)

            finally:
                summary_queue.task_done()

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Summary worker {worker_id} error: {e}")
            # Set shutdown to stop other workers
            shutdown.set()
            raise


# jscpd:ignore-start
# Similar structure to summary_worker but fundamentally different processing
async def embedding_worker(
    worker_id: int,
    embedding_queue: BatchAwareQueue,
    llm_service: LLMService,
    shutdown: asyncio.Event,
    reporter: TelemetryCollector | None = None,
    progress: AsyncProgressWrapper | None = None,
) -> None:
    """Worker that processes embedding generation using batch-aware strategy.
    # jscpd:ignore-end

    Sleeps until a batch is available, processes it, and repeats.
    Uses root node as a natural sentinel to detect completion.

    Args:
        worker_id: ID for logging
        embedding_queue: BatchAwareQueue that coordinates batching
        llm_service: Service for generating embeddings
        shutdown: Event to signal shutdown
    """
    while not shutdown.is_set():
        try:
            # Wait for a batch (sleeps until ready)
            batch = await embedding_queue.get_batch(shutdown)

            if batch is None:
                # Queue is closed, no more work
                logger.debug(f"Embedding worker {worker_id}: Queue closed, exiting")
                break

            # Process the batch
            await _process_embedding_batch(
                batch, llm_service, reporter, progress, worker_id
            )

            # Mark all items as done
            for _ in batch:
                embedding_queue.task_done()

            # Continue processing until queue empties; do not exit early on root
            # BatchAwareQueue will return None when root is queued and queue drains

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Embedding worker {worker_id} error: {e}")
            # Set shutdown to stop other workers
            shutdown.set()
            raise


async def _process_embedding_batch(
    batch: list[TreeNode],
    llm_service: LLMService,
    reporter: TelemetryCollector | None,
    progress: AsyncProgressWrapper | None,
    worker_id: int,
) -> None:
    """Process a batch of embeddings."""
    try:
        # Generate embeddings for batch
        texts = [node.text for node in batch]
        start_time = time.time()
        embeddings = await llm_service._get_embeddings_batch(texts)

        # Store embeddings on domain nodes for later VectorIndex upsert
        for node, embedding in zip(batch, embeddings):
            try:
                node.embedding = list(embedding)  # ensure JSON-serializable list
            except Exception:
                node.embedding = [float(x) for x in embedding]

        # Log embedding batch processing
        heights = [node.height for node in batch]
        batch_type = (
            "root"
            if any(node.is_root() for node in batch)
            else f"h{min(heights)}-{max(heights)}"
        )
        elapsed = time.time() - start_time
        if logger.isEnabledFor(logging.DEBUG):
            total_tokens = sum(tokenizer.count_tokens(node.text) for node in batch)
            logger.debug(
                f"Embedded batch size={len(batch)} type={batch_type} tokens={total_tokens} time={elapsed:.2f}s"
            )
        else:
            logger.debug(
                f"Embedded batch size={len(batch)} type={batch_type} time={elapsed:.2f}s"
            )

        # Track telemetry
        if reporter:
            # Prepare node embeddings data
            node_embeddings = []
            for node in batch:
                # Token counts are only needed for telemetry; compute lazily here
                token_count = tokenizer.count_tokens(node.text)
                node_embeddings.append((node.id, token_count))

            # Record v2 telemetry with per-node tracking
            model = (
                llm_service.config.embedding_model
                if hasattr(llm_service, "config")
                else "unknown"
            )
            reporter.record_embedding_call_v2(
                node_embeddings=node_embeddings,
                batch_size=len(batch),
                model=model,
                start_time=start_time,
            )

        # Update progress for each batch processed (not per node)
        if progress:
            await progress.update(1)  # One batch processed

    except Exception as e:
        logger.error(f"Embedding worker {worker_id} batch processing error: {e}")
        raise


async def build_tree_dataflow(
    chunks: list[str],
    document_id: str,
    llm_service: LLMService,
    target_tokens: int = 200,
    max_summary_concurrency: int = 30,
    max_embedding_concurrency: int = 10,
    embedding_batch_size: int = 100,
    processing_strategy: ProcessingStrategy = ProcessingStrategy.BottomToTop,
    reporter: TelemetryCollector | None = None,
    progress: AsyncProgressWrapper | None = None,
) -> list[TreeNode]:
    """Build tree using dataflow pattern.

    Args:
        chunks: List of text chunks from the document
        document_id: Document ID
        llm_service: Service for generating summaries and embeddings
        target_tokens: Target token count for summaries
        max_summary_concurrency: Maximum concurrent summary workers
        max_embedding_concurrency: Maximum concurrent embedding workers
        embedding_batch_size: Batch size for embeddings
        reporter: Optional telemetry collector
        progress: Optional progress wrapper for updates

    Returns:
        List of TreeNode objects ready for database insertion
    """
    # Create leaf nodes
    lookup, leaves = create_leaf_nodes(chunks, document_id, reporter)

    # Build internal nodes
    build_internal_nodes(lookup, leaves, document_id, reporter)

    # Initialize queues and storage
    summary_queue: asyncio.PriorityQueue[SummaryJob] = asyncio.PriorityQueue()
    embedding_queue = BatchAwareQueue(batch_size=embedding_batch_size)

    # Queue leaf embeddings immediately (they have text)
    for leaf in leaves:
        if leaf.text:  # Leaf nodes always have text
            await embedding_queue.put(leaf)

    # Create shutdown event
    shutdown = asyncio.Event()

    # Start workers
    summary_workers = []
    for i in range(max_summary_concurrency):
        worker = asyncio.create_task(
            summary_worker(
                i,
                lookup,
                summary_queue,
                embedding_queue,
                llm_service,
                shutdown,
                processing_strategy,
                target_tokens,
                reporter,
                progress,
            )
        )
        summary_workers.append(worker)

    embedding_workers = []
    for i in range(max_embedding_concurrency):
        worker = asyncio.create_task(
            embedding_worker(
                i,
                embedding_queue,
                llm_service,
                shutdown,
                reporter,
                progress,
            )
        )
        embedding_workers.append(worker)

    try:
        # Start the cascade - poke all height 1 nodes (parents of leaves)
        # For trees with only leaves (no internal nodes), no summaries needed
        if len(lookup) > len(leaves):
            # Collect unique height-1 nodes and sort by span_start for ordered processing
            height_1_nodes = list(
                set(leaf.parent_id for leaf in leaves if leaf.parent_id)
            )
            height_1_nodes.sort(key=lambda node_id: lookup[node_id].span_start)

            # Poke all unique height 1 nodes in document order - they can all start immediately
            for node_id in height_1_nodes:
                poke(node_id, lookup, summary_queue, processing_strategy)

        # Wait for all summaries to complete
        await summary_queue.join()

        # Wait for all embeddings to complete
        # The root node acts as a natural sentinel, so workers will exit
        # when they process it (always the last node)
        await embedding_queue.join()

        # NOW signal shutdown - all work is complete
        shutdown.set()

        # Wait for all workers to finish cleanly
        all_workers = summary_workers + embedding_workers
        results = await asyncio.gather(*all_workers, return_exceptions=True)

        # Check for any errors
        for result in results:
            if isinstance(result, Exception):
                raise result

    finally:
        # Always clean up workers
        shutdown.set()

        # Cancel all workers
        for worker in summary_workers:
            worker.cancel()
        for worker in embedding_workers:
            worker.cancel()

        await asyncio.gather(
            *(summary_workers + embedding_workers), return_exceptions=True
        )

    # Return all nodes ready for database insertion
    return list(lookup.values())
