"""Tests for indexing engine retry behavior.

These tests verify that transient LLM failures are retried via implicit retry
(next scan finds same eligible pair and recreates the job).
"""

from unittest.mock import patch

import pytest

from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.services.summary_utils import SummaryResult
from tests.conftest import IndexerRuntimeHarness


class TestImplicitRetry:
    """Tests for implicit retry behavior when jobs fail."""

    @pytest.mark.asyncio
    async def test_transient_failure_should_retry(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
        disable_strict_errors: None,  # Retry tests need non-strict mode
    ) -> None:
        """Transient LLM failure should be retried on next scan.

        Expected behavior (design intent):
        1. Create document with 2 leaves (1 pair needing summarization)
        2. LLM fails on FIRST summarization call
        3. Next scan finds same eligible pair → recreates job → retries
        4. LLM succeeds on second call
        5. Tree should have 1 root (parent created on retry)

        Current behavior (bug due to unauthorized mark_failed code):
        - LLM fails → job marked as failed → next scan skips → no retry
        - Tree has 2 roots (orphaned leaves)
        """
        doc_id = "test-retry"

        # Track call count to fail first call, succeed on subsequent
        call_count = 0

        async def mock_summarize_text(
            text: str,
            target_tokens: int,
            *,
            parent_id: str | None = None,
            reporter: object = None,
            prev_context: str | None = None,
            text_tokens: int | None = None,
            summary_system_prompt: str | None = None,
        ) -> SummaryResult:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call fails (simulates transient error)
                raise RuntimeError("Transient LLM error - should be retried")

            # Subsequent calls succeed
            return SummaryResult(
                summary="Test summary",
                retry_count=0,
                summary_tokens=20,
            )

        # Create a document with enough text to create exactly 2 leaves
        # With target_chunk_tokens=50, we need ~100+ tokens worth of text
        # Each word is roughly 1 token, so ~200 words should create 2-4 leaves
        test_text = " ".join([f"Word{i}" for i in range(150)])

        # Patch the LLM service's summarize method
        with patch.object(
            indexer_runtime_harness.llm_service,
            "_summarize_text",
            new=mock_summarize_text,
        ):
            await indexer_runtime_harness.append(
                doc_id,
                test_text,
                replace_existing=True,
                await_idle=True,
            )

        # Check the tree structure
        doc_store = storage_backend.for_document(doc_id)
        all_nodes = list(doc_store.nodes.get_all())

        leaves = [n for n in all_nodes if n.height == 0]
        roots = [n for n in all_nodes if n.parent_id is None]

        # With implicit retry working:
        # - First summarization fails
        # - Next scan recreates the job
        # - Second summarization succeeds
        # - Tree has proper parent node
        # - Should have 1 root (or a valid forest structure)

        leaf_count = len(leaves)
        root_count = len(roots)

        # Calculate expected roots for a valid forest
        # For N leaves, expected roots = popcount(N) (number of 1-bits)
        expected_roots = bin(leaf_count).count("1")

        # The test FAILS if we have more roots than expected
        # because orphaned leaves become extra roots
        assert root_count == expected_roots, (
            f"Expected {expected_roots} root(s) for {leaf_count} leaves, "
            f"but got {root_count}. "
            f"This indicates implicit retry is broken - failed jobs are not being retried."
        )

        # Verify the summarization was actually called multiple times
        # (at least 2: first fail, then success)
        assert call_count >= 2, (
            f"Expected at least 2 summarization calls (fail then succeed), "
            f"but only got {call_count}. Retry mechanism may not be working."
        )

    @pytest.mark.asyncio
    async def test_multiple_failures_eventually_succeed(
        self,
        indexer_runtime_harness: IndexerRuntimeHarness,
        storage_backend: StorageBackend,
        disable_strict_errors: None,  # Retry tests need non-strict mode
    ) -> None:
        """Multiple transient failures should all eventually retry and succeed.

        This tests that even if the first few attempts fail, the system
        keeps retrying until success (via implicit retry on each scan).
        """
        doc_id = "test-multi-retry"

        # Fail first 2 calls, succeed on third
        call_count = 0

        async def mock_summarize_text(
            text: str,
            target_tokens: int,
            *,
            parent_id: str | None = None,
            reporter: object = None,
            prev_context: str | None = None,
            text_tokens: int | None = None,
            summary_system_prompt: str | None = None,
        ) -> SummaryResult:
            nonlocal call_count
            call_count += 1

            if call_count <= 2:
                raise RuntimeError(f"Transient error #{call_count}")

            return SummaryResult(
                summary="Test summary",
                retry_count=0,
                summary_tokens=20,
            )

        test_text = " ".join([f"Word{i}" for i in range(150)])

        with patch.object(
            indexer_runtime_harness.llm_service,
            "_summarize_text",
            new=mock_summarize_text,
        ):
            await indexer_runtime_harness.append(
                doc_id,
                test_text,
                replace_existing=True,
                await_idle=True,
            )

        doc_store = storage_backend.for_document(doc_id)
        all_nodes = list(doc_store.nodes.get_all())
        leaves = [n for n in all_nodes if n.height == 0]
        roots = [n for n in all_nodes if n.parent_id is None]

        leaf_count = len(leaves)
        expected_roots = bin(leaf_count).count("1")
        root_count = len(roots)

        assert root_count == expected_roots, (
            f"Expected {expected_roots} root(s), got {root_count}. "
            f"Multiple retries should eventually succeed."
        )

        assert (
            call_count >= 3
        ), f"Expected at least 3 calls (2 fails + 1 success), got {call_count}."
