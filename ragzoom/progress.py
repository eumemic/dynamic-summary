"""Progress tracking utilities for RagZoom."""

import asyncio
import logging
from typing import Any, Optional

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

logger = logging.getLogger(__name__)


class GlobalProgressTracker:
    """Tracks overall progress across all indexing stages."""

    def __init__(self, total_chunks: int, show_progress: bool = True):
        """Initialize progress tracker.

        Args:
            total_chunks: Number of leaf chunks
            show_progress: Whether to show progress bar
        """
        self.total_chunks = total_chunks
        self.show_progress = show_progress and HAS_TQDM

        # Calculate expected operations
        self.leaf_operations = total_chunks  # Embeddings for leaves
        self.tree_operations = self._estimate_tree_operations(total_chunks)
        self.total_operations = self.leaf_operations + self.tree_operations

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

    def _estimate_tree_operations(self, num_leaves: int) -> int:
        """Estimate number of tree building operations."""
        # Each level has half the nodes of the previous
        # Each node needs 1 summary + 1 embedding
        operations = 0
        level_size = num_leaves

        while level_size > 1:
            level_size = (level_size + 1) // 2
            operations += level_size * 2  # summary + embedding

        return operations

    def update(self, n: int = 1, stage: Optional[str] = None) -> None:
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

    async def update(self, n: int = 1, stage: Optional[str] = None) -> None:
        """Thread-safe async update."""
        async with self.lock:
            self.tracker.update(n, stage)

    def update_sync(self, n: int = 1, stage: Optional[str] = None) -> None:
        """Sync update for non-async contexts."""
        self.tracker.update(n, stage)
