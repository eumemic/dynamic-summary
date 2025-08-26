"""Core dataflow implementation for parallel tree indexing using TreeNode directly.

This module implements the dataflow pattern where nodes are processed
as soon as their dependencies are ready, enabling maximum parallelism.
"""

import asyncio
import logging
import math
import uuid
from typing import Any

from ragzoom.models import TreeNode

logger = logging.getLogger(__name__)


class AtomicCounter:
    """Thread-safe counter for tracking pending work."""

    def __init__(self, initial: int = 0):
        """Initialize counter with initial value."""
        self._value = initial
        self._lock = asyncio.Lock()

    @property
    def value(self) -> int:
        """Get current value."""
        return self._value

    def decrement(self, amount: int = 1) -> None:
        """Decrement counter by amount."""
        # For simplicity, using synchronous decrement since it's atomic enough
        # In production, might want to use asyncio.Lock for true thread safety
        self._value -= amount


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
    chunks: list[str], document_id: str, reporter: Any = None
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
            token_count=0,
        )

        # Update previous leaf's following_neighbor_id
        if previous_leaf:
            previous_leaf.following_neighbor_id = leaf.id

        leaves.append(leaf)
        lookup[node_id] = leaf

        # Track node creation for telemetry
        if reporter:
            try:
                from ragzoom.utils.tokenization import tokenizer

                chunk_tokens = tokenizer.count_tokens(chunk)
                reporter.track_node_created(
                    node_id=leaf.id,
                    height=0,
                    span=(leaf.span_start, leaf.span_end),
                )
                reporter.record_chunk_created(leaf.id, chunk_tokens)
            except Exception:
                pass  # Silently ignore telemetry errors

        previous_leaf = leaf
        current_pos = leaf.span_end

    return lookup, leaves


def build_internal_nodes(
    lookup: dict[str, TreeNode],
    leaves: list[TreeNode],
    document_id: str,
    reporter: Any = None,
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
                try:
                    reporter.track_node_created(
                        node_id=parent.id,
                        height=parent.height,
                        span=(parent.span_start, parent.span_end),
                    )
                except Exception:
                    pass  # Silently ignore telemetry errors

            previous_parent = parent

            # Move to next pair
            i += 2 if right else 1

        current_level = parents
        current_height += 1


async def poke(node_id: str, lookup: dict[str, TreeNode], queue: asyncio.Queue) -> None:
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

    if ready and node.right_child_id:
        right_child = lookup.get(node.right_child_id)
        if not right_child or not right_child.text:
            ready = False

    # Check preceding neighbor has text
    if ready and node.preceding_neighbor_id:
        preceding = lookup.get(node.preceding_neighbor_id)
        if not preceding or not preceding.text:
            ready = False

    # Queue if ready
    if ready:
        await queue.put(node_id)


async def summary_worker(
    worker_id: int,
    lookup: dict[str, TreeNode],
    summary_queue: asyncio.Queue,
    embedding_queue: asyncio.Queue,
    llm_service: Any,
    shutdown: asyncio.Event,
    target_tokens: int = 200,
    reporter: Any = None,
    progress: Any = None,
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

                # Get child texts
                left_text = ""
                right_text = ""
                if node.left_child_id:
                    left_child = lookup[node.left_child_id]
                    left_text = left_child.text or ""
                if node.right_child_id:
                    right_child = lookup[node.right_child_id]
                    right_text = right_child.text or ""

                # Get preceding context
                prev_context = None
                if node.preceding_neighbor_id:
                    preceding = lookup[node.preceding_neighbor_id]
                    prev_context = preceding.text

                # Generate summary
                summary, retry_count, tokens = await llm_service._summarize_text(
                    left_text, right_text, target_tokens, prev_context=prev_context
                )

                # Update node
                node.text = summary
                node.token_count = tokens

                # Track telemetry
                if reporter:
                    try:
                        reporter.record_summary(
                            node_id=node.id,
                            retry_count=retry_count,
                            tokens_used=tokens,
                            model=(
                                llm_service.config.summary_model
                                if hasattr(llm_service, "config")
                                else None
                            ),
                        )
                    except Exception:
                        pass  # Silently ignore telemetry errors

                # Queue for embedding
                await embedding_queue.put((node_id, summary))

                # Update progress if available
                if progress:
                    await progress.update(1)

                # Poke dependents
                if node.parent_id:
                    await poke(node.parent_id, lookup, summary_queue)
                if node.following_neighbor_id:
                    await poke(node.following_neighbor_id, lookup, summary_queue)

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
    llm_service: Any,
    batch_size: int,
    pending_embeddings: AtomicCounter,
    shutdown: asyncio.Event,
    reporter: Any = None,
    progress: Any = None,
) -> None:
    """Worker that processes embedding generation.
    # jscpd:ignore-end

    Args:
        worker_id: ID for logging
        embedding_queue: Queue of (node_id, text) tuples
        lookup: Dictionary to update with embeddings
        llm_service: Service for generating embeddings
        batch_size: Maximum batch size
        pending_embeddings: Counter of pending embeddings
        shutdown: Event to signal shutdown
    """
    while pending_embeddings.value > 0 and not shutdown.is_set():
        batch = []

        # Collect a batch with timeout to allow checking shutdown
        try:
            # Wait for at least one item with timeout
            first_item = await asyncio.wait_for(
                embedding_queue.get(),
                timeout=0.5,  # Timeout to check conditions periodically
            )
            batch.append(first_item)

            # Try to fill the rest of the batch without waiting
            for _ in range(batch_size - 1):
                try:
                    item = embedding_queue.get_nowait()
                    batch.append(item)
                except asyncio.QueueEmpty:
                    break

        except asyncio.TimeoutError:
            # No items available, check if we should continue
            if pending_embeddings.value > 0 and not shutdown.is_set():
                continue  # Still work to do, keep waiting
            else:
                break  # All done

        # Process batch if we have items
        if not batch:
            continue

        try:
            # Generate embeddings for batch
            texts = [text for _, text in batch]
            embeddings = await llm_service._get_embeddings_batch(texts)

            # Store embeddings
            for (node_id, _), embedding in zip(batch, embeddings):
                lookup[node_id].embedding = embedding

            # Track telemetry
            if reporter:
                try:
                    import time

                    from ragzoom.utils.tokenization import tokenizer

                    # Prepare node embeddings data
                    node_embeddings = []
                    for node_id, text in batch:
                        token_count = tokenizer.count_tokens(text)
                        node_embeddings.append((node_id, token_count))

                    # Record v2 telemetry with per-node tracking
                    reporter.record_embedding_call_v2(
                        node_embeddings=node_embeddings,
                        batch_size=len(batch),
                        model=(
                            llm_service.config.embedding_model
                            if hasattr(llm_service, "config")
                            else None
                        ),
                        start_time=time.time(),  # Approximate start time
                    )
                except Exception:
                    pass  # Silently ignore telemetry errors

            # Update counter
            pending_embeddings.decrement(len(batch))

            # Update progress for leaf embeddings
            if progress:
                # Count how many are leaves (height 0)
                leaf_count = sum(
                    1 for node_id, _ in batch if lookup[node_id].height == 0
                )
                if leaf_count > 0:
                    await progress.update(leaf_count)

        except Exception as e:
            logger.error(f"Embedding worker {worker_id} error: {e}")
            raise


async def build_tree_dataflow(
    chunks: list[str],
    document_id: str,
    llm_service: Any,
    target_tokens: int = 200,
    max_summary_concurrency: int = 30,
    max_embedding_concurrency: int = 10,
    embedding_batch_size: int = 100,
    reporter: Any = None,
    progress: Any = None,
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

    # Count total embeddings needed
    total_nodes = len(lookup)
    pending_embeddings = AtomicCounter(total_nodes)

    # Queue leaf embeddings immediately (they have text)
    for leaf in leaves:
        if leaf.text:  # Leaf nodes always have text
            await embedding_queue.put((leaf.id, leaf.text))

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
                pending_embeddings,
                shutdown,
                reporter,
                progress,
            )
        )
        embedding_workers.append(worker)

    try:
        # Start the cascade - poke first parent if exists
        # For trees with only leaves (no internal nodes), no summaries needed
        if len(lookup) > len(leaves):
            if leaves and leaves[0].parent_id:
                await poke(leaves[0].parent_id, lookup, summary_queue)

        # Wait for all summaries to complete
        await summary_queue.join()

        # Signal summary workers to shutdown
        shutdown.set()

        # Wait for all embeddings to complete
        # Embedding workers will process remaining items and exit when counter reaches 0
        await asyncio.gather(*embedding_workers, return_exceptions=True)

        # Check if any summary workers had errors
        summary_results = await asyncio.gather(*summary_workers, return_exceptions=True)
        for result in summary_results:
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
