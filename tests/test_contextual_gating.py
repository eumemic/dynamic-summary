"""Tests for contextual indexing gating logic.

The gating logic ensures nodes are only processed when preceding context is available.
Key concepts:
- Dynamic summary frontier: computed by summing root tokens until exceeding budget B
- eligible_span_start: computed by walking leaves from frontier, accumulating up to K tokens
- Left children with span_start <= eligible_span_start are eligible for building parents
- No sibling eligibility checks - if left is eligible, proceed with both siblings
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.server.worker_coordinator import WorkerCoordinator


@pytest.fixture
def mock_llm_service() -> MagicMock:
    """Create a mock LLM service."""
    service = MagicMock()
    service._summarize_text = MagicMock()
    service.embed_texts = MagicMock()
    return service


@pytest.fixture
def index_config_with_lag() -> IndexConfig:
    """Create an IndexConfig with context_lag_tokens set."""
    return IndexConfig(
        target_chunk_tokens=200,
        preceding_summary_budget_tokens=2000,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
        retry_threshold=0.2,
        max_retries=0,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
        context_lag_tokens=500,
    )


@pytest.fixture
def index_config_no_lag() -> IndexConfig:
    """Create an IndexConfig with context_lag_tokens=0 (strictest gating)."""
    return IndexConfig(
        target_chunk_tokens=200,
        preceding_summary_budget_tokens=2000,
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
        retry_threshold=0.2,
        max_retries=0,
        embedding_batch_size=100,
        use_anti_verbatim_vaccine=True,
        processing_strategy="bottom_to_top",
        context_lag_tokens=0,
    )


@pytest.fixture
def operational_config() -> OperationalConfig:
    """Create operational configuration."""
    return OperationalConfig(
        openai_api_key=SecretStr("test-key"),
        backend="sqlite",
        database_url="sqlite:///:memory:",
    )


class TestComputeEligibleSpan:
    """Tests for _compute_eligible_span method."""

    def test_eligible_span_with_no_leaves(
        self,
        storage_backend: StorageBackend,
        index_config_with_lag: IndexConfig,
        operational_config: OperationalConfig,
        mock_llm_service: MagicMock,
    ) -> None:
        """With no leaves beyond frontier, eligible_span_end equals frontier."""
        coordinator = WorkerCoordinator(
            store=storage_backend,
            index_config=index_config_with_lag,
            operational_config=operational_config,
            llm_service=mock_llm_service,
        )

        # Empty document - frontier is 0, no leaves
        frontier, eligible_end, last_eligible_start = (
            coordinator._compute_eligible_span("doc1")
        )
        assert frontier == 0
        assert eligible_end == 0
        assert last_eligible_start == 0

    def test_eligible_span_with_zero_k(
        self,
        storage_backend: StorageBackend,
        index_config_no_lag: IndexConfig,
        operational_config: OperationalConfig,
        mock_llm_service: MagicMock,
    ) -> None:
        """With K=0, eligible_span_end equals frontier (no leaves eligible)."""
        coordinator = WorkerCoordinator(
            store=storage_backend,
            index_config=index_config_no_lag,  # K=0
            operational_config=operational_config,
            llm_service=mock_llm_service,
        )

        frontier, eligible_end, last_eligible_start = (
            coordinator._compute_eligible_span("doc1")
        )
        assert frontier == 0
        assert eligible_end == 0  # K=0 means no leaves beyond frontier are eligible
        assert last_eligible_start == 0


class TestCheckContextualReadiness:
    """Tests for _check_contextual_readiness method."""

    def test_node_at_frontier_is_ready(
        self,
        storage_backend: StorageBackend,
        index_config_with_lag: IndexConfig,
        operational_config: OperationalConfig,
        mock_llm_service: MagicMock,
    ) -> None:
        """Left children at or before eligible_span_start are ready."""
        coordinator = WorkerCoordinator(
            store=storage_backend,
            index_config=index_config_with_lag,
            operational_config=operational_config,
            llm_service=mock_llm_service,
        )

        # With empty document, frontier=0, eligible_span_start=0
        # Left child with span_start=0 should be ready
        assert coordinator._check_contextual_readiness(0, "doc1") is True

    def test_node_beyond_eligible_span_is_blocked(
        self,
        storage_backend: StorageBackend,
        index_config_no_lag: IndexConfig,
        operational_config: OperationalConfig,
        mock_llm_service: MagicMock,
    ) -> None:
        """Left children with span_start > eligible_span_start are blocked."""
        coordinator = WorkerCoordinator(
            store=storage_backend,
            index_config=index_config_no_lag,  # K=0
            operational_config=operational_config,
            llm_service=mock_llm_service,
        )

        # With K=0 and empty document, eligible_span_start=0
        # Left child with span_start=100 should be blocked
        assert coordinator._check_contextual_readiness(100, "doc1") is False


class TestEligibleSpanCaching:
    """Tests for eligible span caching behavior."""

    def test_eligible_span_is_cached(
        self,
        storage_backend: StorageBackend,
        index_config_with_lag: IndexConfig,
        operational_config: OperationalConfig,
        mock_llm_service: MagicMock,
    ) -> None:
        """Eligible span is cached after first computation."""
        coordinator = WorkerCoordinator(
            store=storage_backend,
            index_config=index_config_with_lag,
            operational_config=operational_config,
            llm_service=mock_llm_service,
        )

        # First call computes and caches
        result1 = coordinator._get_eligible_span_start("doc1")
        assert "doc1" in coordinator._eligible_span_ends

        # Second call uses cache
        result2 = coordinator._get_eligible_span_start("doc1")
        assert result1 == result2

    def test_frontier_invalidation_clears_eligible_span_cache(
        self,
        storage_backend: StorageBackend,
        index_config_with_lag: IndexConfig,
        operational_config: OperationalConfig,
        mock_llm_service: MagicMock,
    ) -> None:
        """Invalidating frontier also clears eligible span cache."""
        coordinator = WorkerCoordinator(
            store=storage_backend,
            index_config=index_config_with_lag,
            operational_config=operational_config,
            llm_service=mock_llm_service,
        )

        # Populate caches
        coordinator._get_eligible_span_start("doc1")
        coordinator.compute_dynamic_summary_frontier("doc1")

        assert "doc1" in coordinator._eligible_span_ends
        assert "doc1" in coordinator._tree_frontiers

        # Invalidate frontier
        coordinator.invalidate_tree_frontier("doc1")

        # Both caches should be cleared
        assert "doc1" not in coordinator._eligible_span_ends
        assert "doc1" not in coordinator._tree_frontiers


class TestGatingIntegration:
    """Integration tests for gating with _resolve_dependencies."""

    def test_coordinator_has_gating_methods(
        self,
        storage_backend: StorageBackend,
        index_config_no_lag: IndexConfig,
        operational_config: OperationalConfig,
        mock_llm_service: MagicMock,
    ) -> None:
        """Verify the coordinator has all required gating methods."""
        coordinator = WorkerCoordinator(
            store=storage_backend,
            index_config=index_config_no_lag,
            operational_config=operational_config,
            llm_service=mock_llm_service,
        )

        assert hasattr(coordinator, "_compute_eligible_span")
        assert hasattr(coordinator, "_get_eligible_span_start")
        assert hasattr(coordinator, "_check_contextual_readiness")
        assert coordinator._index_config.context_lag_tokens == 0
