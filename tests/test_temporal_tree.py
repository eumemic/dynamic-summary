"""Tests for temporal metadata propagation through the tree structure."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore


@pytest.fixture
def doc_store(storage_backend: StorageBackend) -> Generator[DocumentStore, None, None]:
    """Create a document store for testing temporal tree propagation."""
    document_id = "temporal-test-doc"
    storage_backend.clear_document(document_id)
    store = storage_backend.for_document(document_id)
    store.set_metadata(
        file_path="temporal-test.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )
    yield store
    storage_backend.clear_document(document_id)


def _make_node_data(
    node_id: str,
    text: str,
    span_start: int,
    span_end: int,
    height: int,
    level_index: int,
    document_id: str,
    *,
    time_start: float | None = None,
    time_end: float | None = None,
    left_child_id: str | None = None,
    right_child_id: str | None = None,
    parent_id: str | None = None,
) -> NodeDataDict:
    """Create node data with temporal fields."""
    return {
        "node_id": node_id,
        "text": text,
        "span_start": span_start,
        "span_end": span_end,
        "token_count": len(text.split()),
        "height": height,
        "level_index": level_index,
        "document_id": document_id,
        "time_start": time_start,
        "time_end": time_end,
        "left_child_id": left_child_id,
        "right_child_id": right_child_id,
        "parent_id": parent_id,
        "preceding_neighbor_id": None,
        "following_neighbor_id": None,
    }


class TestInnerNodeTimestampPropagation:
    """Test that inner nodes inherit timestamps from their children.

    The spec requires:
    - Inner nodes have time_start = left_child.time_start
    - Inner nodes have time_end = right_child.time_end

    This mirrors how span_start and span_end propagate.
    """

    def test_inner_node_has_correct_timestamps_when_children_have_timestamps(
        self, doc_store: DocumentStore
    ) -> None:
        """Inner node should inherit time_start from left child, time_end from right child."""
        document_id = "temporal-test-doc"

        # Create two leaf nodes with timestamps
        left_leaf = _make_node_data(
            node_id="leaf-left",
            text="First chunk of content",
            span_start=0,
            span_end=100,
            height=0,
            level_index=0,
            document_id=document_id,
            time_start=1705848600.0,  # 2024-01-21T14:30:00Z
            time_end=1705848612.0,  # 2024-01-21T14:30:12Z
        )

        right_leaf = _make_node_data(
            node_id="leaf-right",
            text="Second chunk of content",
            span_start=100,
            span_end=200,
            height=0,
            level_index=1,
            document_id=document_id,
            time_start=1705848620.0,  # 2024-01-21T14:30:20Z
            time_end=1705848650.0,  # 2024-01-21T14:30:50Z
        )

        # Create parent node that should inherit timestamps from children
        # time_start should come from left child, time_end from right child
        parent_node = _make_node_data(
            node_id="parent",
            text="Summary of both chunks",
            span_start=0,  # from left
            span_end=200,  # from right
            height=1,
            level_index=0,
            document_id=document_id,
            time_start=1705848600.0,  # Expected: left_child.time_start
            time_end=1705848650.0,  # Expected: right_child.time_end
            left_child_id="leaf-left",
            right_child_id="leaf-right",
        )

        # Add nodes to database
        doc_store.nodes.add_batch([left_leaf, right_leaf, parent_node])
        doc_store.nodes.update_parent_references_batch(
            [("leaf-left", "parent"), ("leaf-right", "parent")]
        )

        # Verify parent node has correct timestamps
        parent = doc_store.nodes.get("parent")
        assert parent is not None
        assert getattr(parent, "time_start", None) == 1705848600.0
        assert getattr(parent, "time_end", None) == 1705848650.0

    def test_inner_node_timestamps_are_none_when_children_have_none(
        self, doc_store: DocumentStore
    ) -> None:
        """Inner node should have None timestamps when children have None."""
        document_id = "temporal-test-doc"

        # Create two leaf nodes without timestamps
        left_leaf = _make_node_data(
            node_id="leaf-left-no-ts",
            text="First chunk without timestamp",
            span_start=0,
            span_end=100,
            height=0,
            level_index=0,
            document_id=document_id,
            time_start=None,
            time_end=None,
        )

        right_leaf = _make_node_data(
            node_id="leaf-right-no-ts",
            text="Second chunk without timestamp",
            span_start=100,
            span_end=200,
            height=0,
            level_index=1,
            document_id=document_id,
            time_start=None,
            time_end=None,
        )

        # Parent should also have None timestamps
        parent_node = _make_node_data(
            node_id="parent-no-ts",
            text="Summary without timestamps",
            span_start=0,
            span_end=200,
            height=1,
            level_index=0,
            document_id=document_id,
            time_start=None,
            time_end=None,
            left_child_id="leaf-left-no-ts",
            right_child_id="leaf-right-no-ts",
        )

        doc_store.nodes.add_batch([left_leaf, right_leaf, parent_node])

        parent = doc_store.nodes.get("parent-no-ts")
        assert parent is not None
        assert getattr(parent, "time_start", None) is None
        assert getattr(parent, "time_end", None) is None

    def test_deep_tree_timestamp_propagation(self, doc_store: DocumentStore) -> None:
        """Timestamps should propagate correctly through multiple tree levels.

        Tree structure:
                    root (h=2, time_start=T1, time_end=T4)
                   /    \
              P1 (h=1)   P2 (h=1)
             /   \\       /   \
           L1    L2    L3    L4
          (T1)  (T2)  (T3)  (T4)
        """
        document_id = "temporal-test-doc"

        # Create 4 leaves with sequential timestamps
        leaves = [
            _make_node_data(
                node_id="L1",
                text="Leaf 1",
                span_start=0,
                span_end=50,
                height=0,
                level_index=0,
                document_id=document_id,
                time_start=1705848600.0,  # T1 start
                time_end=1705848610.0,  # T1 end
            ),
            _make_node_data(
                node_id="L2",
                text="Leaf 2",
                span_start=50,
                span_end=100,
                height=0,
                level_index=1,
                document_id=document_id,
                time_start=1705848620.0,  # T2 start
                time_end=1705848630.0,  # T2 end
            ),
            _make_node_data(
                node_id="L3",
                text="Leaf 3",
                span_start=100,
                span_end=150,
                height=0,
                level_index=2,
                document_id=document_id,
                time_start=1705848640.0,  # T3 start
                time_end=1705848650.0,  # T3 end
            ),
            _make_node_data(
                node_id="L4",
                text="Leaf 4",
                span_start=150,
                span_end=200,
                height=0,
                level_index=3,
                document_id=document_id,
                time_start=1705848660.0,  # T4 start
                time_end=1705848670.0,  # T4 end
            ),
        ]

        # Create height-1 parents
        # P1: covers L1 and L2, so time_start=L1.time_start, time_end=L2.time_end
        p1 = _make_node_data(
            node_id="P1",
            text="Parent 1",
            span_start=0,
            span_end=100,
            height=1,
            level_index=0,
            document_id=document_id,
            time_start=1705848600.0,  # L1.time_start
            time_end=1705848630.0,  # L2.time_end
            left_child_id="L1",
            right_child_id="L2",
        )

        # P2: covers L3 and L4, so time_start=L3.time_start, time_end=L4.time_end
        p2 = _make_node_data(
            node_id="P2",
            text="Parent 2",
            span_start=100,
            span_end=200,
            height=1,
            level_index=1,
            document_id=document_id,
            time_start=1705848640.0,  # L3.time_start
            time_end=1705848670.0,  # L4.time_end
            left_child_id="L3",
            right_child_id="L4",
        )

        # Root: covers P1 and P2, so time_start=P1.time_start, time_end=P2.time_end
        root = _make_node_data(
            node_id="root",
            text="Root",
            span_start=0,
            span_end=200,
            height=2,
            level_index=0,
            document_id=document_id,
            time_start=1705848600.0,  # P1.time_start = L1.time_start
            time_end=1705848670.0,  # P2.time_end = L4.time_end
            left_child_id="P1",
            right_child_id="P2",
        )

        # Add all nodes
        doc_store.nodes.add_batch(leaves + [p1, p2, root])
        doc_store.nodes.update_parent_references_batch(
            [
                ("L1", "P1"),
                ("L2", "P1"),
                ("L3", "P2"),
                ("L4", "P2"),
                ("P1", "root"),
                ("P2", "root"),
            ]
        )

        # Verify root timestamps span the entire time range
        root_node = doc_store.nodes.get("root")
        assert root_node is not None
        assert getattr(root_node, "time_start", None) == 1705848600.0  # L1's start
        assert getattr(root_node, "time_end", None) == 1705848670.0  # L4's end

        # Verify intermediate parents
        p1_node = doc_store.nodes.get("P1")
        assert p1_node is not None
        assert getattr(p1_node, "time_start", None) == 1705848600.0  # L1's start
        assert getattr(p1_node, "time_end", None) == 1705848630.0  # L2's end

        p2_node = doc_store.nodes.get("P2")
        assert p2_node is not None
        assert getattr(p2_node, "time_start", None) == 1705848640.0  # L3's start
        assert getattr(p2_node, "time_end", None) == 1705848670.0  # L4's end


@pytest.mark.usefixtures("sqlite_backend")
class TestRuntimeTimestampPropagation:
    """Test that _summarize_pair() correctly propagates timestamps at runtime.

    These tests verify the actual indexing engine behavior, not just database storage.
    """

    @pytest.mark.asyncio
    async def test_inner_node_timestamp_propagation(
        self, storage_backend: StorageBackend
    ) -> None:
        """When indexing engine creates inner nodes, timestamps propagate from children.

        This test:
        1. Creates leaf nodes with timestamps directly in the database
        2. Triggers summarization via the indexing engine
        3. Verifies the resulting inner node has correct timestamps
        """
        from unittest.mock import AsyncMock, MagicMock, patch

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob
        from ragzoom.services.llm_service import LLMService
        from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

        document_id = "runtime-temporal-test"
        storage_backend.clear_document(document_id)
        store = storage_backend.for_document(document_id)
        store.set_metadata(
            file_path="runtime-temporal.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Create two leaf nodes with timestamps
        left_leaf = _make_node_data(
            node_id="left-leaf",
            text="First chunk content for summarization",
            span_start=0,
            span_end=100,
            height=0,
            level_index=0,
            document_id=document_id,
            time_start=1705848600.0,  # 2024-01-21T14:30:00Z
            time_end=1705848612.0,  # 2024-01-21T14:30:12Z
        )

        right_leaf = _make_node_data(
            node_id="right-leaf",
            text="Second chunk content for summarization",
            span_start=100,
            span_end=200,
            height=0,
            level_index=1,
            document_id=document_id,
            time_start=1705848620.0,  # 2024-01-21T14:30:20Z
            time_end=1705848650.0,  # 2024-01-21T14:30:50Z
        )

        store.nodes.add_batch([left_leaf, right_leaf])

        # Create minimal IndexingEngine with mocked LLM
        index_config = IndexConfig.load(target_chunk_tokens=100)

        # Mock LLM service
        llm_service = MagicMock(spec=LLMService)
        llm_service._summarize_text = AsyncMock(
            return_value=SummaryResult(
                summary="Combined summary of both chunks",
                retry_count=0,
                summary_tokens=50,
                usage=AccumulatedUsage(prompt_tokens=100, completion_tokens=50),
            )
        )

        # Mock OpenAI client for retriever
        mock_openai = MagicMock()
        mock_openai.embeddings.create = MagicMock(
            return_value=MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536)],
                usage=MagicMock(prompt_tokens=10, total_tokens=10),
            )
        )

        indexing_engine = IndexingEngine(
            store=storage_backend,
            llm_service=llm_service,
            index_config=index_config,
            openai_client=mock_openai,
        )

        # Create a summary job for the two leaf nodes
        job = SummaryJob(
            document_id=document_id,
            left_id="left-leaf",
            right_id="right-leaf",
        )

        # Execute the summarization
        with patch.object(
            indexing_engine, "_run_context_for_node", return_value=(None, None)
        ):
            await indexing_engine._summarize_pair(job)

        # Find the parent node by checking the left leaf's parent_id
        left_node = store.nodes.get("left-leaf")
        assert left_node is not None
        parent_id = getattr(left_node, "parent_id", None)
        assert (
            parent_id is not None
        ), "Left leaf should have a parent after summarization"

        # Fetch the created parent node from the database
        parent_node = store.nodes.get(parent_id)
        assert parent_node is not None

        # Key assertion: timestamps should be propagated from children
        assert (
            getattr(parent_node, "time_start", None) == 1705848600.0
        )  # left.time_start
        assert getattr(parent_node, "time_end", None) == 1705848650.0  # right.time_end

        # Verify span is also correct (existing behavior)
        assert getattr(parent_node, "span_start", None) == 0  # left.span_start
        assert getattr(parent_node, "span_end", None) == 200  # right.span_end

        # Cleanup
        storage_backend.clear_document(document_id)

    @pytest.mark.asyncio
    async def test_inner_node_none_timestamps_when_children_have_none(
        self, storage_backend: StorageBackend
    ) -> None:
        """When children have no timestamps, inner node should have None timestamps."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from ragzoom.config import IndexConfig
        from ragzoom.server.indexing_engine import IndexingEngine, SummaryJob
        from ragzoom.services.llm_service import LLMService
        from ragzoom.services.summary_utils import AccumulatedUsage, SummaryResult

        document_id = "runtime-no-temporal-test"
        storage_backend.clear_document(document_id)
        store = storage_backend.for_document(document_id)
        store.set_metadata(
            file_path="runtime-no-temporal.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Create two leaf nodes WITHOUT timestamps
        left_leaf = _make_node_data(
            node_id="left-leaf-no-ts",
            text="First chunk without timestamp",
            span_start=0,
            span_end=100,
            height=0,
            level_index=0,
            document_id=document_id,
            time_start=None,
            time_end=None,
        )

        right_leaf = _make_node_data(
            node_id="right-leaf-no-ts",
            text="Second chunk without timestamp",
            span_start=100,
            span_end=200,
            height=0,
            level_index=1,
            document_id=document_id,
            time_start=None,
            time_end=None,
        )

        store.nodes.add_batch([left_leaf, right_leaf])

        index_config = IndexConfig.load(target_chunk_tokens=100)

        llm_service = MagicMock(spec=LLMService)
        llm_service._summarize_text = AsyncMock(
            return_value=SummaryResult(
                summary="Combined summary",
                retry_count=0,
                summary_tokens=50,
                usage=AccumulatedUsage(prompt_tokens=100, completion_tokens=50),
            )
        )

        mock_openai = MagicMock()
        mock_openai.embeddings.create = MagicMock(
            return_value=MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536)],
                usage=MagicMock(prompt_tokens=10, total_tokens=10),
            )
        )

        indexing_engine = IndexingEngine(
            store=storage_backend,
            llm_service=llm_service,
            index_config=index_config,
            openai_client=mock_openai,
        )

        job = SummaryJob(
            document_id=document_id,
            left_id="left-leaf-no-ts",
            right_id="right-leaf-no-ts",
        )

        with patch.object(
            indexing_engine, "_run_context_for_node", return_value=(None, None)
        ):
            await indexing_engine._summarize_pair(job)

        # Find the parent node by checking the left leaf's parent_id
        left_node = store.nodes.get("left-leaf-no-ts")
        assert left_node is not None
        parent_id = getattr(left_node, "parent_id", None)
        assert parent_id is not None

        parent_node = store.nodes.get(parent_id)
        assert parent_node is not None

        # Timestamps should be None when children have None
        assert getattr(parent_node, "time_start", None) is None
        assert getattr(parent_node, "time_end", None) is None

        storage_backend.clear_document(document_id)
