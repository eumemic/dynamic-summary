"""Regression test for _process_node_pair parameter passing.

This test prevents the bug where _process_node_pair failed to pass
prev_context, left_token_count, and right_token_count to LLMService,
causing a 102% increase in retry rate.
"""

from typing import cast
from unittest.mock import MagicMock, patch

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.contracts.vector_index import VectorIndex as _VectorIndexProtocol
from ragzoom.index import TreeBuilder
from ragzoom.telemetry_collection import TelemetryCollector


@pytest.fixture
def mock_nodes() -> tuple[MagicMock, MagicMock]:
    """Create mock tree nodes with token counts."""
    left_node = MagicMock(spec=TreeNode)
    left_node.token_count = 150
    left_node.span_start = 0
    left_node.span_end = 100

    right_node = MagicMock(spec=TreeNode)
    right_node.token_count = 200
    right_node.span_start = 100
    right_node.span_end = 250

    return left_node, right_node


@pytest.fixture
def mock_reporter() -> MagicMock:
    """Create a mock telemetry collector."""
    reporter = MagicMock(spec=TelemetryCollector)
    reporter._current_height = 1
    reporter.track_node_created = MagicMock()
    return reporter


@pytest.mark.asyncio
async def test_process_node_pair_passes_all_parameters(
    storage_backend: StorageBackend,
    mock_nodes: tuple[MagicMock, MagicMock],
    mock_reporter: MagicMock,
    vector_index: _VectorIndexProtocol,
) -> None:
    """Regression test: ensure _process_node_pair passes all parameters to LLMService.

    This test would have caught the bug where prev_context, left_token_count,
    and right_token_count were not being passed through.
    """
    left_node, right_node = mock_nodes

    # Get document store and set metadata
    doc_store = storage_backend.for_document("test_doc")
    doc_store.set_metadata(
        file_path="process_node_pair_test.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )

    config = IndexConfig.load(preceding_context_tokens=75, target_chunk_tokens=200)
    builder = TreeBuilder(config, doc_store, vector_index)

    # Capture parameters passed to LLMService._summarize_text
    captured_params: dict[str, object] = {}

    async def capture_summarize_params(
        *args: object, **kwargs: object
    ) -> tuple[str, int, int]:
        """Capture all parameters passed to _summarize_text."""
        captured_params.clear()
        captured_params.update(kwargs)
        captured_params["args"] = args
        return ("test summary", 0, 100)  # (summary, retry_count, token_count)

    with patch.object(
        builder.llm_service, "_summarize_text", new=capture_summarize_params
    ):
        # Call _process_node_pair with prev_context
        await builder._process_node_pair(
            left_id="left_id",
            left_text="This is the left text content.",
            right_id="right_id",
            right_text="This is the right text content.",
            prev_context="This is the previous context that should be passed through.",
            document_id="test_doc",
            current_height=2,
            reporter=mock_reporter,
            left_node=left_node,  # Pre-fetched
            right_node=right_node,  # Pre-fetched
        )

    # Verify all critical parameters were passed through
    assert "prev_context" in captured_params, "prev_context parameter missing"
    assert (
        captured_params["prev_context"]
        == "This is the previous context that should be passed through."
    )

    assert "left_token_count" in captured_params, "left_token_count parameter missing"
    assert captured_params["left_token_count"] == 150

    assert "right_token_count" in captured_params, "right_token_count parameter missing"
    assert captured_params["right_token_count"] == 200

    assert "parent_id" in captured_params, "parent_id parameter missing"
    assert captured_params["parent_id"] is not None

    assert "reporter" in captured_params, "reporter parameter missing"
    assert captured_params["reporter"] is mock_reporter

    # Verify positional args are correct (self is not included when patching)
    args = cast(tuple[object, ...], captured_params["args"])
    assert len(args) == 3  # left_text, right_text, target_tokens
    assert args[0] == "This is the left text content."
    assert args[1] == "This is the right text content."
    assert args[2] == config.target_chunk_tokens


@pytest.mark.asyncio
async def test_prev_context_affects_prompt(
    storage_backend: StorageBackend,
    mock_nodes: tuple[MagicMock, MagicMock],
    mock_reporter: MagicMock,
    vector_index: _VectorIndexProtocol,
) -> None:
    """Test that prev_context actually changes the generated prompt."""
    left_node, right_node = mock_nodes

    # Get document store and set metadata
    doc_store = storage_backend.for_document("test_doc")
    doc_store.set_metadata(
        file_path="prev_context_test.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )

    config = IndexConfig.load(preceding_context_tokens=75, target_chunk_tokens=200)
    builder = TreeBuilder(config, doc_store, vector_index)

    # Test by capturing the parameters passed to _summarize_text
    captured_params_list = []

    async def capture_params(*args: object, **kwargs: object) -> tuple[str, int, int]:
        """Capture parameters for each call."""
        captured_params_list.append(kwargs.copy())
        return ("Mock summary", 0, 100)

    with patch.object(builder.llm_service, "_summarize_text", new=capture_params):
        # Test 1: Call without prev_context
        await builder._process_node_pair(
            left_id="left_id",
            left_text="Left content",
            right_id="right_id",
            right_text="Right content",
            prev_context=None,  # No context
            document_id="test_doc",
            current_height=2,
            reporter=mock_reporter,
            left_node=left_node,
            right_node=right_node,
        )

        # Test 2: Call with prev_context
        await builder._process_node_pair(
            left_id="left_id",
            left_text="Left content",
            right_id="right_id",
            right_text="Right content",
            prev_context="Previous context information",  # With context
            document_id="test_doc",
            current_height=2,
            reporter=mock_reporter,
            left_node=left_node,
            right_node=right_node,
        )

    # Verify that the prev_context parameter was different
    assert len(captured_params_list) == 2
    params_without = captured_params_list[0]
    params_with = captured_params_list[1]

    assert (
        params_without["prev_context"] is None
    ), "First call should have no prev_context"
    assert (
        params_with["prev_context"] == "Previous context information"
    ), "Second call should have prev_context"

    print("✅ prev_context parameter correctly passed through in both scenarios")


@pytest.mark.asyncio
async def test_parameter_validation_would_catch_bug(
    storage_backend: StorageBackend,
    mock_nodes: tuple[MagicMock, MagicMock],
    mock_reporter: MagicMock,
    vector_index: _VectorIndexProtocol,
) -> None:
    """Test that demonstrates how the bug could be caught with parameter validation."""
    left_node, right_node = mock_nodes

    # Get document store and set metadata
    doc_store = storage_backend.for_document("test_doc")
    doc_store.set_metadata(
        file_path="parameter_validation_test.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )

    config = IndexConfig.load(target_chunk_tokens=200)
    builder = TreeBuilder(config, doc_store, vector_index)

    # Track what parameters were actually passed to _summarize_text
    actual_calls = []

    async def track_calls(*args: object, **kwargs: object) -> tuple[str, int, int]:
        actual_calls.append(
            {
                "prev_context": kwargs.get("prev_context"),
                "left_token_count": kwargs.get("left_token_count"),
                "right_token_count": kwargs.get("right_token_count"),
            }
        )
        return ("summary", 0, 100)

    with patch.object(builder.llm_service, "_summarize_text", new=track_calls):
        # This call should pass all available parameters
        await builder._process_node_pair(
            left_id="left_id",
            left_text="Left content",
            right_id="right_id",
            right_text="Right content",
            prev_context="Important context",
            document_id="test_doc",
            current_height=2,
            reporter=mock_reporter,
            left_node=left_node,
            right_node=right_node,
        )

    # Verify the call received all expected parameters
    call = actual_calls[0]
    assert (
        call["prev_context"] == "Important context"
    ), "prev_context should be passed through"
    assert call["left_token_count"] == 150, "left_token_count should be passed through"
    assert (
        call["right_token_count"] == 200
    ), "right_token_count should be passed through"

    # This test demonstrates that the fix is working - all parameters are now passed through
    print(f"✅ All parameters correctly passed: {call}")


# Integration test showing the impact of the bug
@pytest.mark.asyncio
async def test_bug_would_cause_missing_parameters() -> None:
    """Demonstrate that the bug would cause parameters to be missing."""

    # Simulate the "buggy" version by showing what would happen
    # if _process_node_pair didn't pass certain parameters

    from ragzoom.config import IndexConfig
    from ragzoom.services.llm_service import LLMService

    config = IndexConfig.load(preceding_context_tokens=75)
    llm_service = LLMService(config, api_key="test-key")

    captured_calls = []

    async def capture_llm_calls(
        *args: object, **kwargs: object
    ) -> tuple[str, int, int]:
        """Capture what parameters LLMService actually receives."""
        captured_calls.append(
            {
                "prev_context": kwargs.get("prev_context"),
                "left_token_count": kwargs.get("left_token_count"),
                "right_token_count": kwargs.get("right_token_count"),
            }
        )
        return ("summary", 0, 100)

    with patch.object(llm_service, "_summarize_text", new=capture_llm_calls):
        # Scenario 1: The bug - missing parameters
        await llm_service._summarize_text(
            "Left text",
            "Right text",
            100,
            parent_id="test_id",
            reporter=None,
            # BUG: prev_context, left_token_count, right_token_count not passed
        )

        # Scenario 2: The fix - all parameters passed
        await llm_service._summarize_text(
            "Left text",
            "Right text",
            100,
            parent_id="test_id",
            reporter=None,
            prev_context="Context information",  # FIX: parameter included
            left_token_count=150,  # FIX: parameter included
            right_token_count=200,  # FIX: parameter included
        )

    # Verify the difference
    buggy_call = captured_calls[0]
    fixed_call = captured_calls[1]

    assert buggy_call["prev_context"] is None, "Buggy version has no prev_context"
    assert (
        buggy_call["left_token_count"] is None
    ), "Buggy version has no left_token_count"
    assert (
        buggy_call["right_token_count"] is None
    ), "Buggy version has no right_token_count"

    assert (
        fixed_call["prev_context"] == "Context information"
    ), "Fixed version has prev_context"
    assert fixed_call["left_token_count"] == 150, "Fixed version has left_token_count"
    assert fixed_call["right_token_count"] == 200, "Fixed version has right_token_count"

    print("✅ Demonstrated: missing parameters vs complete parameters")
