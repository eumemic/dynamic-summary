"""Tests for progress tracking functionality."""

import asyncio
from unittest.mock import Mock, patch

import pytest

from ragzoom.progress import AsyncProgressWrapper, GlobalProgressTracker


class TestGlobalProgressTracker:
    """Test the global progress tracker."""

    def test_init_with_tqdm(self):
        """Test initialization with tqdm available."""
        with patch("ragzoom.progress.HAS_TQDM", True):
            with patch("ragzoom.progress.tqdm") as mock_tqdm:
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                tracker = GlobalProgressTracker(
                    10, show_progress=True, embedding_batch_size=100
                )

                assert tracker.total_chunks == 10
                assert tracker.embedding_batch_size == 100
                assert tracker.show_progress is True
                assert tracker.total_operations > 0
                assert tracker.pbar == mock_pbar

                # Verify tqdm was called with correct parameters
                mock_tqdm.assert_called_once()
                call_args = mock_tqdm.call_args[1]
                assert call_args["total"] == tracker.total_operations
                assert call_args["unit"] == " ops"
                assert call_args["leave"] is False

    def test_init_without_tqdm(self):
        """Test initialization when tqdm is not available."""
        with patch("ragzoom.progress.HAS_TQDM", False):
            tracker = GlobalProgressTracker(10, show_progress=True)

            assert tracker.total_chunks == 10
            assert tracker.show_progress is False
            assert tracker.pbar is None

    def test_init_no_progress(self):
        """Test initialization with progress disabled."""
        tracker = GlobalProgressTracker(10, show_progress=False)

        assert tracker.show_progress is False
        assert tracker.pbar is None

    def test_calculate_total_operations(self):
        """Test total operation calculation with new formula."""
        # Test with batch_size=100
        tracker = GlobalProgressTracker(1, embedding_batch_size=100)
        # 1 chunk = 0 internal nodes + 1 embedding batch = 1 total
        assert tracker._calculate_total_operations(1) == 1

        tracker = GlobalProgressTracker(2, embedding_batch_size=100)
        # 2 chunks = 1 internal node + 2 embedding batches (2 non-root + 1 root) = 3 total
        assert tracker._calculate_total_operations(2) == 3

        tracker = GlobalProgressTracker(4, embedding_batch_size=100)
        # 4 chunks = 3 internal nodes + 2 embedding batches (6 non-root + 1 root) = 5 total
        assert tracker._calculate_total_operations(4) == 5

        tracker = GlobalProgressTracker(100, embedding_batch_size=100)
        # 100 chunks = 102 internal nodes + 4 embedding batches (201 non-root, 1 root) = 106 total
        assert tracker._calculate_total_operations(100) == 106

        tracker = GlobalProgressTracker(731, embedding_batch_size=100)
        # 731 chunks = 734 internal nodes + 16 embedding batches = 750 total
        assert tracker._calculate_total_operations(731) == 750

    def test_update_with_progress(self):
        """Test updating progress with tqdm."""
        with patch("ragzoom.progress.HAS_TQDM", True):
            with patch("ragzoom.progress.tqdm") as mock_tqdm:
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                tracker = GlobalProgressTracker(10, show_progress=True)
                tracker.update(5)

                assert tracker.current == 5
                mock_pbar.update.assert_called_once_with(5)

                tracker.update(3, stage="tree")
                assert tracker.current == 8
                assert mock_pbar.update.call_count == 2

    def test_update_without_progress(self):
        """Test updating progress without tqdm."""
        tracker = GlobalProgressTracker(10, show_progress=False)

        tracker.update(5)
        assert tracker.current == 5

        tracker.update(3)
        assert tracker.current == 8

    def test_close(self):
        """Test closing progress bar."""
        with patch("ragzoom.progress.HAS_TQDM", True):
            with patch("ragzoom.progress.tqdm") as mock_tqdm:
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                tracker = GlobalProgressTracker(10, show_progress=True)
                tracker.close()

                mock_pbar.close.assert_called_once()

    def test_context_manager(self):
        """Test context manager support."""
        with patch("ragzoom.progress.HAS_TQDM", True):
            with patch("ragzoom.progress.tqdm") as mock_tqdm:
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                with GlobalProgressTracker(10, show_progress=True) as tracker:
                    assert tracker.pbar == mock_pbar

                # Verify close was called on exit
                mock_pbar.close.assert_called_once()


class TestAsyncProgressWrapper:
    """Test the async progress wrapper."""

    @pytest.mark.asyncio
    async def test_async_update(self):
        """Test async update method."""
        mock_tracker = Mock()
        wrapper = AsyncProgressWrapper(mock_tracker)

        await wrapper.update(5)
        mock_tracker.update.assert_called_once_with(5, None)

        await wrapper.update(3, stage="tree")
        assert mock_tracker.update.call_count == 2
        mock_tracker.update.assert_called_with(3, "tree")

    def test_sync_update(self):
        """Test sync update method."""
        mock_tracker = Mock()
        wrapper = AsyncProgressWrapper(mock_tracker)

        wrapper.update_sync(5)
        mock_tracker.update.assert_called_once_with(5, None)

        wrapper.update_sync(3, stage="tree")
        assert mock_tracker.update.call_count == 2
        mock_tracker.update.assert_called_with(3, "tree")

    @pytest.mark.asyncio
    async def test_concurrent_updates(self):
        """Test that concurrent updates are properly synchronized."""
        mock_tracker = Mock()
        wrapper = AsyncProgressWrapper(mock_tracker)

        # Simulate concurrent updates
        async def update_task(n):
            await wrapper.update(n)

        # Run multiple updates concurrently
        await asyncio.gather(
            update_task(1),
            update_task(2),
            update_task(3),
            update_task(4),
            update_task(5),
        )

        # All updates should have been called
        assert mock_tracker.update.call_count == 5

        # Check that all values were passed
        called_values = [call[0][0] for call in mock_tracker.update.call_args_list]
        assert sorted(called_values) == [1, 2, 3, 4, 5]


class TestProgressIntegration:
    """Test progress tracking integration scenarios."""

    def test_full_indexing_scenario(self):
        """Test a full indexing scenario with progress tracking."""
        with patch("ragzoom.progress.HAS_TQDM", True):
            with patch("ragzoom.progress.tqdm") as mock_tqdm:
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                # Simulate indexing 100 chunks
                with GlobalProgressTracker(100, show_progress=True) as tracker:
                    # Process leaves
                    for _ in range(100):
                        tracker.update(1, stage="leaves")

                    # Process tree levels
                    # Level 1: 50 nodes
                    for _ in range(50):
                        tracker.update(2, stage="tree")  # summary + embedding

                    # Level 2: 25 nodes
                    for _ in range(25):
                        tracker.update(2, stage="tree")

                    # Continue until root...

                # Verify progress bar was closed
                mock_pbar.close.assert_called_once()

    def test_error_handling(self):
        """Test that progress bar is closed even on error."""
        with patch("ragzoom.progress.HAS_TQDM", True):
            with patch("ragzoom.progress.tqdm") as mock_tqdm:
                mock_pbar = Mock()
                mock_tqdm.return_value = mock_pbar

                try:
                    with GlobalProgressTracker(10, show_progress=True) as tracker:
                        tracker.update(5)
                        raise ValueError("Test error")
                except ValueError:
                    pass

                # Progress bar should still be closed
                mock_pbar.close.assert_called_once()
