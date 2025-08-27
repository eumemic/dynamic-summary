"""Progress tracking utilities for RagZoom."""

import asyncio
import logging
from typing import Any

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

logger = logging.getLogger(__name__)


class GlobalProgressTracker:
    """Tracks overall progress across all indexing stages."""

    def __init__(
        self,
        total_chunks: int,
        show_progress: bool = True,
        embedding_batch_size: int = 100,
    ):
        """Initialize progress tracker.

        Args:
            total_chunks: Number of leaf chunks
            show_progress: Whether to show progress bar
            embedding_batch_size: Size of embedding batches for progress calculation
        """
        self.total_chunks = total_chunks
        self.embedding_batch_size = embedding_batch_size
        self.show_progress = show_progress and HAS_TQDM

        # Calculate expected operations
        self.total_operations = self._calculate_total_operations(total_chunks)

        # Create progress bar
        if self.show_progress:
            # Use simpler format to avoid display issues
            self.pbar = tqdm(
                total=self.total_operations,
                unit=" ops",
                leave=False,  # Don't leave the bar on screen when done
                smoothing=0.3,  # Smooth out time estimates
                miniters=1,  # Update on every iteration
                # Simpler format without descriptions to avoid formatting issues
                bar_format="{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
            )
        else:
            self.pbar = None

        self.current = 0
        self.stage = "leaves"
        self.start_time = None

    def _calculate_total_operations(self, num_leaves: int) -> int:
        """Calculate total number of progress operations.

        Progress units represent:
        - One summary generation for each internal node
        - One embedding batch (multiple nodes processed together)
        """
        import math

        # Calculate number of internal nodes (all non-leaf nodes)
        internal_nodes = 0
        level_size = num_leaves
        while level_size > 1:
            level_size = (level_size + 1) // 2
            internal_nodes += level_size

        # Calculate total nodes
        total_nodes = num_leaves + internal_nodes

        # Calculate number of embedding batches
        # Root gets its own batch, all others are batched together
        if total_nodes == 1:
            embedding_batches = 1
        else:
            embedding_batches = (
                math.ceil((total_nodes - 1) / self.embedding_batch_size) + 1
            )

        # Total operations = summaries + embedding batches
        return internal_nodes + embedding_batches

    def update(self, n: int = 1, stage: str | None = None) -> None:
        """Update progress."""
        self.current += n
        if self.pbar:
            self.pbar.update(n)

    def close(self) -> None:
        """Close progress bar."""
        if self.pbar:
            self.pbar.close()

    def __enter__(self) -> "GlobalProgressTracker":
        """Context manager support."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Close on exit."""
        self.close()


class AsyncProgressWrapper:
    """Wrapper to make progress updates work in async context."""

    def __init__(self, tracker: GlobalProgressTracker):
        self.tracker = tracker
        self.lock = asyncio.Lock()

    async def update(self, n: int = 1, stage: str | None = None) -> None:
        """Thread-safe async update."""
        async with self.lock:
            self.tracker.update(n, stage)

    def update_sync(self, n: int = 1, stage: str | None = None) -> None:
        """Sync update for non-async contexts."""
        self.tracker.update(n, stage)
