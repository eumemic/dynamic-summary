"""Integration test for retry metrics collection."""

import json

from ragzoom.metrics import IndexingMetrics, IndexingMetricsReporter, RetryStats


class TestRetryMetricsIntegration:
    """Test retry metrics collection in the indexing pipeline."""

    def test_retry_stats_dataclass(self):
        """Test RetryStats basic functionality."""
        stats = RetryStats()

        # Add some retry counts
        stats.add_retry(0)  # No retries
        stats.add_retry(2)  # 2 retries
        stats.add_retry(1)  # 1 retry
        stats.add_retry(3)  # 3 retries

        assert stats.count == 4
        assert stats.total_retries == 6
        assert stats.avg_retries == 1.5
        assert stats.min_retries == 0
        assert stats.max_retries == 3

        # Test serialization
        data = stats.to_dict()
        assert data["count"] == 4
        assert data["total_retries"] == 6
        assert data["avg_retries"] == 1.5
        assert data["min_retries"] == 0
        assert data["max_retries"] == 3

    def test_indexing_metrics_retry_properties(self):
        """Test IndexingMetrics retry-related properties."""
        metrics = IndexingMetrics(
            start_time=0,
            end_time=10,
            source_document_tokens=1000,
            chunks_created=10,
            embedding_cost_per_1k=0.001,
            summary_input_cost_per_1k=0.01,
            summary_output_cost_per_1k=0.02,
        )

        # Add retry stats for different levels
        metrics.retry_stats[0] = RetryStats()  # Leaf level
        metrics.retry_stats[0].add_retry(0)
        metrics.retry_stats[0].add_retry(0)

        metrics.retry_stats[1] = RetryStats()  # Level 1
        metrics.retry_stats[1].add_retry(1)
        metrics.retry_stats[1].add_retry(2)
        metrics.retry_stats[1].add_retry(1)

        metrics.retry_stats[2] = RetryStats()  # Level 2
        metrics.retry_stats[2].add_retry(3)

        # Test computed properties
        assert metrics.total_retries == 7  # 0+0+1+2+1+3
        assert metrics.retries_per_1k_tokens == 7.0  # 7 retries / 1K tokens
        assert metrics.avg_retries_per_summary == 7 / 6  # 7 retries / 6 summaries

    def test_reporter_retry_tracking(self):
        """Test IndexingMetricsReporter retry tracking."""
        from ragzoom.config import RagZoomConfig

        config = RagZoomConfig(
            openai_api_key="test",
            embedding_cost_per_1k=0.001,
            summary_input_cost_per_1k=0.01,
            summary_output_cost_per_1k=0.02,
        )

        reporter = IndexingMetricsReporter("test-doc", 1000, config)

        # Record some summaries with retries
        reporter.record_summary_retry(0, 0)  # Level 0, no retries
        reporter.record_summary_retry(1, 2)  # Level 1, 2 retries
        reporter.record_summary_retry(1, 1)  # Level 1, 1 retry
        reporter.record_summary_retry(2, 3)  # Level 2, 3 retries

        # Finalize and check
        metrics = reporter.finalize()

        assert len(metrics.retry_stats) == 3  # 3 levels
        assert metrics.retry_stats[0].count == 1
        assert metrics.retry_stats[0].total_retries == 0
        assert metrics.retry_stats[1].count == 2
        assert metrics.retry_stats[1].total_retries == 3
        assert metrics.retry_stats[2].count == 1
        assert metrics.retry_stats[2].total_retries == 3

        assert metrics.total_retries == 6
        assert metrics.retries_per_1k_tokens == 6.0

    def test_metrics_json_serialization_with_retries(self):
        """Test that retry stats are included in JSON export."""
        from ragzoom.config import RagZoomConfig

        config = RagZoomConfig(
            openai_api_key="test",
            embedding_cost_per_1k=0.001,
            summary_input_cost_per_1k=0.01,
            summary_output_cost_per_1k=0.02,
        )

        reporter = IndexingMetricsReporter("test-doc", 1000, config)

        # Add some retry data
        reporter.record_summary_retry(0, 1)
        reporter.record_summary_retry(1, 2)

        metrics = reporter.finalize()

        # Convert to dict
        data = metrics.to_dict()

        # Check retry data is included
        assert "retry_stats" in data
        assert "0" in data["retry_stats"]
        assert "1" in data["retry_stats"]

        assert "retry_summary" in data
        assert data["retry_summary"]["total_retries"] == 3
        assert data["retry_summary"]["retries_per_1k_tokens"] == 3.0

        # Ensure it's JSON serializable
        json_str = json.dumps(data)
        assert json_str  # Should not raise
