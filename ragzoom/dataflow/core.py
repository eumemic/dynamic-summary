"""Core dataflow implementation for parallel tree indexing using TreeNode directly.

This module implements the dataflow pattern where nodes are processed
as soon as their dependencies are ready, enabling maximum parallelism.
"""

import asyncio
import logging
import math
import time
import uuid

from ragzoom.models import TreeNode
from ragzoom.progress import AsyncProgressWrapper
from ragzoom.services.llm_service import LLMService
from ragzoom.telemetry_collection import TelemetryCollector
from ragzoom.utils.tokenization import tokenizer

logger = logging.getLogger(__name__)


def _generate_node_id() -> str:
    """Generate a unique node ID."""
    return str(uuid.uuid4())


def _calculate_tree_depth(num_leaves: int) -> int:
    """Calculate the depth of the tree for binary path encoding."""
    if num_leaves <= 1:
        return 1
    return math.ceil(math.log2(num_leaves))


def _generate_leaf_path(index: int, tree_depth: int) -> str:
    """Generate binary path for a leaf node."""
    return format(index, f"0{tree_depth}b")


def _derive_parent_path(child_path: str) -> str:
    """Derive parent path by removing last bit from child path."""
    return child_path[:-1] if child_path else ""


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

    # Calculate tree depth for path generation
    tree_depth = _calculate_tree_depth(len(chunks))

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
            path=_generate_leaf_path(i, tree_depth),
            document_id=document_id,
            # No parent/children for leaves initially
            parent_id=None,
            left_child_id=None,
            right_child_id=None,
            # Neighbor relationships
            preceding_neighbor_id=previous_leaf.id if previous_leaf else None,
            following_neighbor_id=None,  # Will be set by next leaf
            # Leaves need embeddings but we'll generate them later
            embedding=[],  # Empty list for now
            token_count=chunk_tokens,  # Set actual token count
        )

        # Update previous leaf's following_neighbor_id
        if previous_leaf:
            previous_leaf.following_neighbor_id = leaf.id

        leaves.append(leaf)
        lookup[node_id] = leaf

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
                path=_derive_parent_path(left.path),
                document_id=document_id,
                # Children
                parent_id=None,  # Will be set when grandparent created
                left_child_id=left.id,
                right_child_id=right.id if right else None,
                # Neighbors
                preceding_neighbor_id=previous_parent.id if previous_parent else None,
                following_neighbor_id=None,  # Will be set by next parent
                # Empty embedding until generated
                embedding=[],
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


def poke(node_id: str, lookup: dict[str, TreeNode], queue: asyncio.Queue) -> None:
    """Check if node's dependencies are ready and queue if so.

    Args:
        node_id: ID of node to check
        lookup: Dictionary containing all nodes
        queue: Queue to add node to if ready
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

    # Queue if ready
    if ready:
        queue.put_nowait(node_id)


async def summary_worker(
    worker_id: int,
    lookup: dict[str, TreeNode],
    summary_queue: asyncio.Queue,
    embedding_queue: asyncio.Queue,
    llm_service: LLMService,
    shutdown: asyncio.Event,
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
                node_id = await asyncio.wait_for(summary_queue.get(), timeout=0.1)
            except asyncio.TimeoutError:
                continue

            try:
                node = lookup[node_id]

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
                    parent_id=node_id,  # Pass node_id as parent_id for telemetry
                    reporter=reporter,  # Pass reporter so telemetry works
                    prev_context=prev_context,
                    left_token_count=left_token_count,
                    right_token_count=right_token_count,
                )

                # Update node
                node.text = summary
                node.token_count = tokens

                # Poke dependents IMMEDIATELY (no yielding between set and poke!)
                if node.parent_id:
                    poke(node.parent_id, lookup, summary_queue)

                # Only right children poke their following neighbor's parent
                # (Left children's following neighbor shares the same parent)
                if node.is_right_child() and node.following_neighbor_id:
                    following_neighbor = lookup.get(node.following_neighbor_id)
                    if following_neighbor and following_neighbor.parent_id:
                        poke(following_neighbor.parent_id, lookup, summary_queue)

                # Queue for embedding (use put_nowait - we have unlimited queue)
                embedding_queue.put_nowait((node_id, summary))

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
    embedding_queue: asyncio.Queue,
    lookup: dict[str, TreeNode],
    llm_service: LLMService,
    batch_size: int,
    shutdown: asyncio.Event,
    embedding_complete: asyncio.Event,
    reporter: TelemetryCollector | None = None,
    progress: AsyncProgressWrapper | None = None,
) -> None:
    """Worker that processes embedding generation using batch-aware strategy.
    # jscpd:ignore-end

    Only processes when there are enough items to make a full batch,
    maximizing API efficiency by avoiding single-item calls.

    Args:
        worker_id: ID for logging
        embedding_queue: Queue of (node_id, text) tuples
        lookup: Dictionary to update with embeddings
        llm_service: Service for generating embeddings
        batch_size: Target batch size for optimal API utilization
        shutdown: Event to signal shutdown
        embedding_complete: Event to signal no more items will be added
    """
    while not shutdown.is_set():
        try:
            # Check if we have enough items for a full batch
            if embedding_queue.qsize() >= batch_size:
                # Grab a full batch immediately
                batch = []
                for _ in range(batch_size):
                    item = await embedding_queue.get()
                    batch.append(item)

                # Process full batch
                await _process_embedding_batch(
                    batch, lookup, llm_service, reporter, progress, worker_id
                )

                # Mark all items as done
                for _ in batch:
                    embedding_queue.task_done()

            elif embedding_complete.is_set() and not embedding_queue.empty():
                # All summaries are done but we have a partial batch - process it
                batch = []
                while not embedding_queue.empty():
                    try:
                        item = embedding_queue.get_nowait()
                        batch.append(item)
                    except asyncio.QueueEmpty:
                        break

                if batch:
                    # Process partial batch
                    await _process_embedding_batch(
                        batch, lookup, llm_service, reporter, progress, worker_id
                    )

                    # Mark all items as done
                    for _ in batch:
                        embedding_queue.task_done()

                # No more work to do
                break

            else:
                # Not enough items for full batch and not complete yet - wait briefly
                await asyncio.sleep(0.01)  # Small sleep to avoid busy-waiting

        except Exception as e:
            logger.error(f"Embedding worker {worker_id} error: {e}")
            raise


async def _process_embedding_batch(
    batch: list[tuple[str, str]],
    lookup: dict[str, TreeNode],
    llm_service: LLMService,
    reporter: TelemetryCollector | None,
    progress: AsyncProgressWrapper | None,
    worker_id: int,
) -> None:
    """Process a batch of embeddings."""
    try:
        # Generate embeddings for batch
        texts = [text for _, text in batch]
        start_time = time.time()
        embeddings = await llm_service._get_embeddings_batch(texts)

        # Store embeddings
        for (node_id, _), embedding in zip(batch, embeddings):
            lookup[node_id].embedding = embedding

        # Track telemetry
        if reporter:
            # Prepare node embeddings data
            node_embeddings = []
            for node_id, text in batch:
                token_count = tokenizer.count_tokens(text)
                node_embeddings.append((node_id, token_count))

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

        # Update progress for leaf embeddings
        if progress:
            # Count how many are leaves (height 0)
            leaf_count = sum(1 for node_id, _ in batch if lookup[node_id].height == 0)
            if leaf_count > 0:
                await progress.update(leaf_count)

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
    summary_queue: asyncio.Queue[str] = asyncio.Queue()
    embedding_queue: asyncio.Queue[tuple[str, str]] = asyncio.Queue()

    # Queue leaf embeddings immediately (they have text)
    for leaf in leaves:
        if leaf.text:  # Leaf nodes always have text
            await embedding_queue.put((leaf.id, leaf.text))

    # Create shutdown and completion events
    shutdown = asyncio.Event()
    embedding_complete = asyncio.Event()

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
                lookup,
                llm_service,
                embedding_batch_size,
                shutdown,
                embedding_complete,
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
                poke(node_id, lookup, summary_queue)

        # Wait for all summaries to complete
        await summary_queue.join()

        # Signal that no more embeddings will be added
        embedding_complete.set()

        # Wait for all embeddings to complete
        # This ensures all embeddings are processed before shutdown
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
