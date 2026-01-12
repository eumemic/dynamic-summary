"""Tests for Phase 6 validation updates for client-managed chunking.

This module tests that validation code properly handles target_chunk_tokens=None
and validates that target_chunk_tokens > 0 before using in calculations.
"""

from __future__ import annotations

import pytest

from ragzoom.contracts.storage_backend import StorageBackend


def test_postgres_validation_handles_none_target(
    storage_backend: StorageBackend,
) -> None:
    """Test that PostgreSQL validation doesn't fail when target_chunk_tokens=None.

    When target_chunk_tokens is None (client-managed chunking mode), the leaf chunk
    size validation should be skipped entirely. This test verifies that validation
    runs successfully without errors.
    """
    # Skip test if not using PostgreSQL backend
    pytest.importorskip("psycopg2")

    # Check if we're using PostgreSQL by trying to access PostgreSQL-specific features
    document_id = "test-validation-none"
    doc_store = storage_backend.for_document(document_id)

    # Check if run_validation_queries is available
    if not hasattr(doc_store.nodes, "run_validation_queries"):
        pytest.skip("Storage backend does not support run_validation_queries")

    # Create a simple tree structure
    doc_store.nodes.add_node(
        node_id="leaf-1",
        text="leaf content",
        embedding=[0.1] * 10,
        span_start=0,
        span_end=100,
        token_count=25,  # Doesn't matter for None target
        height=0,
        level_index=0,
    )

    # Run validation with target_chunk_tokens=None
    result = doc_store.nodes.run_validation_queries(
        document_id=document_id,
        target_chunk_tokens=None,
    )

    # Should succeed without leaf chunk size validation errors
    assert result is not None
    assert "leaf_chunk_size" not in result.checks_run

    # Clean up
    storage_backend.clear_document(document_id)


def test_postgres_validation_handles_zero_target(
    storage_backend: StorageBackend,
) -> None:
    """Test that PostgreSQL validation handles target_chunk_tokens=0 gracefully.

    When target_chunk_tokens is 0, validation should either skip the chunk size
    check or handle it without causing division errors. This is a defensive coding
    check - in practice, configs should validate that target_chunk_tokens > 0 when
    not None.
    """
    # Skip test if not using PostgreSQL backend
    pytest.importorskip("psycopg2")

    document_id = "test-validation-zero"
    doc_store = storage_backend.for_document(document_id)

    # Check if run_validation_queries is available
    if not hasattr(doc_store.nodes, "run_validation_queries"):
        pytest.skip("Storage backend does not support run_validation_queries")

    # Create a leaf node
    doc_store.nodes.add_node(
        node_id="leaf-1",
        text="leaf content",
        embedding=[0.1] * 10,
        span_start=0,
        span_end=100,
        token_count=25,
        height=0,
        level_index=0,
    )

    # Run validation with target_chunk_tokens=0
    # This should not crash or cause division by zero
    result = doc_store.nodes.run_validation_queries(
        document_id=document_id,
        target_chunk_tokens=0,
    )

    # Should succeed without errors - validation should be skipped for 0
    assert result is not None
    assert "leaf_chunk_size" not in result.checks_run

    # Clean up
    storage_backend.clear_document(document_id)


def test_postgres_validation_handles_negative_target(
    storage_backend: StorageBackend,
) -> None:
    """Test that PostgreSQL validation handles negative target_chunk_tokens gracefully.

    This is a defensive coding check - configs should validate positive values,
    but validation queries should handle edge cases without crashing.
    """
    # Skip test if not using PostgreSQL backend
    pytest.importorskip("psycopg2")

    document_id = "test-validation-negative"
    doc_store = storage_backend.for_document(document_id)

    # Check if run_validation_queries is available
    if not hasattr(doc_store.nodes, "run_validation_queries"):
        pytest.skip("Storage backend does not support run_validation_queries")

    # Create a leaf node
    doc_store.nodes.add_node(
        node_id="leaf-1",
        text="leaf content",
        embedding=[0.1] * 10,
        span_start=0,
        span_end=100,
        token_count=25,
        height=0,
        level_index=0,
    )

    # Run validation with target_chunk_tokens=-10
    # This should not crash
    result = doc_store.nodes.run_validation_queries(
        document_id=document_id,
        target_chunk_tokens=-10,
    )

    # Should succeed without errors - validation should be skipped for negative
    assert result is not None
    assert "leaf_chunk_size" not in result.checks_run

    # Clean up
    storage_backend.clear_document(document_id)


def test_postgres_validation_runs_chunk_size_check_for_positive_target(
    storage_backend: StorageBackend,
) -> None:
    """Test that PostgreSQL validation runs chunk size check for positive values.

    When target_chunk_tokens is a positive integer, the validation should run
    the leaf chunk size check as normal.
    """
    # Skip test if not using PostgreSQL backend
    pytest.importorskip("psycopg2")

    document_id = "test-validation-positive"
    doc_store = storage_backend.for_document(document_id)

    # Check if run_validation_queries is available
    if not hasattr(doc_store.nodes, "run_validation_queries"):
        pytest.skip("Storage backend does not support run_validation_queries")

    # Create a leaf node with token_count that matches target
    doc_store.nodes.add_node(
        node_id="leaf-1",
        text="leaf content",
        embedding=[0.1] * 10,
        span_start=0,
        span_end=100,
        token_count=200,  # Matches target
        height=0,
        level_index=0,
    )

    # Run validation with target_chunk_tokens=200
    result = doc_store.nodes.run_validation_queries(
        document_id=document_id,
        target_chunk_tokens=200,
    )

    # Should succeed and leaf_chunk_size check should be run
    assert result is not None
    assert "leaf_chunk_size" in result.checks_run
    # result.leaf_chunk_size could be None or empty list
    if result.leaf_chunk_size is not None:
        assert len(result.leaf_chunk_size) == 0  # No violations

    # Clean up
    storage_backend.clear_document(document_id)
