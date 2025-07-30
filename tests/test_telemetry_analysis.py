"""Test telemetry analysis functions."""

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry import (
    TelemetryAnalysisError,
    analyze_retry_patterns,
    compute_amplification_metrics,
    compute_batch_efficiency,
    compute_metrics_from_telemetry,
    parse_telemetry_format,
)


class TestTelemetryFormatParsing:
    """Test telemetry format parsing."""

    def test_parse_valid_telemetry_format(self) -> None:
        """Test parsing valid telemetry format."""
        telemetry_data = {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "nodes": [
                        {
                            "node_id": "node-1",
                            "node_type": "leaf",
                            "level": 0,
                            "span": [0, 100],
                            "created_at": 1234567890.0,
                        }
                    ]
                }
            },
        }

        result = parse_telemetry_format(telemetry_data)
        assert result["format_version"] == "1.0"
        assert "test_doc" in result["documents"]

    def test_parse_missing_format_version(self) -> None:
        """Test parsing telemetry without format version."""
        telemetry_data = {"documents": {}}

        with pytest.raises(TelemetryAnalysisError, match="Missing format_version"):
            parse_telemetry_format(telemetry_data)

    def test_parse_unsupported_format_version(self) -> None:
        """Test parsing telemetry with unsupported format version."""
        telemetry_data = {"format_version": "2.0", "documents": {}}

        with pytest.raises(
            TelemetryAnalysisError, match="Unsupported telemetry format version"
        ):
            parse_telemetry_format(telemetry_data)

    def test_parse_invalid_data_type(self) -> None:
        """Test parsing non-dictionary data."""
        with pytest.raises(TelemetryAnalysisError, match="must be a dictionary"):
            parse_telemetry_format("invalid")

    def test_parse_invalid_documents_structure(self) -> None:
        """Test parsing with invalid documents structure."""
        telemetry_data = {"format_version": "1.0", "documents": "invalid"}

        with pytest.raises(TelemetryAnalysisError, match="Invalid documents structure"):
            parse_telemetry_format(telemetry_data)


class TestAmplificationMetrics:
    """Test amplification metrics computation."""

    @pytest.fixture
    def config(self) -> RagZoomConfig:
        """Create test config."""
        return RagZoomConfig(
            openai_api_key="test-key",
            summary_input_cost_per_1k=0.0025,  # gpt-4o-mini
            summary_output_cost_per_1k=0.01,
        )

    @pytest.fixture
    def sample_telemetry(self) -> dict:
        """Create sample telemetry data with summary attempts."""
        return {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "nodes": [
                        {
                            "node_id": "summary-1",
                            "node_type": "summary",
                            "level": 1,
                            "span": [0, 200],
                            "created_at": 1234567890.0,
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "target_tokens": 100,
                                    "input_text_tokens": 200,
                                    "prompt_tokens": 250,  # 1.25x input amplification
                                    "completion_tokens": 90,
                                    "actual_tokens": 90,
                                    "status": "accepted",
                                    "model": "gpt-4o-mini",
                                    "timestamp": 1234567891.0,
                                }
                            ],
                        },
                        {
                            "node_id": "summary-2",
                            "node_type": "summary",
                            "level": 1,
                            "span": [200, 400],
                            "created_at": 1234567892.0,
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "target_tokens": 100,
                                    "input_text_tokens": 200,
                                    "prompt_tokens": 300,  # 1.5x input amplification
                                    "completion_tokens": 110,
                                    "actual_tokens": 100,
                                    "status": "accepted",
                                    "model": "gpt-4o-mini",
                                    "timestamp": 1234567893.0,
                                }
                            ],
                        },
                    ]
                }
            },
        }

    def test_compute_amplification_metrics(
        self, config: RagZoomConfig, sample_telemetry: dict
    ) -> None:
        """Test computing amplification metrics from telemetry."""
        result = compute_amplification_metrics(sample_telemetry, config)

        # Should have computed metrics
        assert result["median_cost"] > 0
        assert result["median_input"] > 0
        assert result["median_output"] > 0

        # Check input amplification: median of [1.25, 1.5] = 1.375
        assert result["median_input"] == pytest.approx(1.375, rel=0.01)

        # Check output amplification: median of [1.0, 1.1] = 1.05
        assert result["median_output"] == pytest.approx(1.05, rel=0.01)

        # Check by-level breakdown
        assert 1 in result["by_level"]
        assert len(result["by_level"][1]["input"]) == 2
        assert len(result["by_level"][1]["output"]) == 2
        assert len(result["by_level"][1]["cost"]) == 2

    def test_amplification_metrics_empty_data(self, config: RagZoomConfig) -> None:
        """Test amplification metrics with empty telemetry."""
        empty_telemetry = {"format_version": "1.0", "documents": {}}

        result = compute_amplification_metrics(empty_telemetry, config)

        # Should return zero values
        assert result["median_cost"] == 0.0
        assert result["median_input"] == 0.0
        assert result["median_output"] == 0.0
        assert result["by_level"] == {}

    def test_amplification_metrics_only_leaf_nodes(self, config: RagZoomConfig) -> None:
        """Test amplification metrics with only leaf nodes (no summaries)."""
        leaf_only_telemetry = {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "nodes": [
                        {
                            "node_id": "leaf-1",
                            "node_type": "leaf",
                            "level": 0,
                            "span": [0, 100],
                            "created_at": 1234567890.0,
                        }
                    ]
                }
            },
        }

        result = compute_amplification_metrics(leaf_only_telemetry, config)

        # Should return zero values (no summary attempts)
        assert result["median_cost"] == 0.0
        assert result["median_input"] == 0.0
        assert result["median_output"] == 0.0


class TestBatchEfficiency:
    """Test batch efficiency analysis."""

    def test_compute_batch_efficiency(self) -> None:
        """Test computing batch efficiency from telemetry."""
        telemetry_data = {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "nodes": [
                        {
                            "node_id": "node-1",
                            "node_type": "leaf",
                            "level": 0,
                            "embedding": {
                                "text_tokens": 50,
                                "batch_size": 3,
                                "batch_position": 0,
                                "model": "text-embedding-3-small",
                                "timestamp": 1234567890.0,
                            },
                        },
                        {
                            "node_id": "node-2",
                            "node_type": "leaf",
                            "level": 0,
                            "embedding": {
                                "text_tokens": 45,
                                "batch_size": 3,
                                "batch_position": 1,
                                "model": "text-embedding-3-small",
                                "timestamp": 1234567890.0,  # Same batch
                            },
                        },
                        {
                            "node_id": "node-3",
                            "node_type": "leaf",
                            "level": 0,
                            "embedding": {
                                "text_tokens": 55,
                                "batch_size": 2,
                                "batch_position": 0,
                                "model": "text-embedding-3-small",
                                "timestamp": 1234567891.0,  # Different batch
                            },
                        },
                    ]
                }
            },
        }

        result = compute_batch_efficiency(telemetry_data)

        # Should have 2 unique batches (sizes 3 and 2)
        assert result["total_batches"] == 2
        assert result["total_embeddings"] == 3
        assert result["batch_sizes"] == [3, 2]
        assert result["avg_batch_size"] == 2.5
        # Utilization now uses 95th percentile: 2.5 / 2.95 * 100 ≈ 84.75%
        # (95th percentile of [2, 3] = 2.95)
        assert result["batch_utilization"] == pytest.approx(84.75, rel=0.01)

    def test_batch_efficiency_empty_data(self) -> None:
        """Test batch efficiency with empty telemetry."""
        empty_telemetry = {"format_version": "1.0", "documents": {}}

        result = compute_batch_efficiency(empty_telemetry)

        assert result["total_batches"] == 0
        assert result["total_embeddings"] == 0
        assert result["batch_sizes"] == []
        assert result["avg_batch_size"] == 0.0
        assert result["batch_utilization"] == 0.0


class TestRetryAnalysis:
    """Test retry pattern analysis."""

    def test_analyze_retry_patterns(self) -> None:
        """Test analyzing retry patterns from telemetry."""
        telemetry_data = {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "nodes": [
                        {
                            "node_id": "summary-1",
                            "node_type": "summary",
                            "level": 1,
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "status": "rejected_over",
                                    "rejection_reason": "30% over target",
                                },
                                {
                                    "is_retry": True,
                                    "status": "accepted",
                                },
                            ],
                        },
                        {
                            "node_id": "summary-2",
                            "node_type": "summary",
                            "level": 1,
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "status": "accepted",
                                }
                            ],
                        },
                        {
                            "node_id": "summary-3",
                            "node_type": "summary",
                            "level": 1,
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "status": "rejected_under",
                                    "rejection_reason": "20% under target",
                                },
                                {
                                    "is_retry": True,
                                    "status": "error",
                                    "rejection_reason": "API timeout",
                                },
                            ],
                        },
                    ]
                }
            },
        }

        result = analyze_retry_patterns(telemetry_data)

        # 3 summary nodes total, 2 needed retries
        assert result["total_nodes_with_summaries"] == 3
        assert result["nodes_with_retries"] == 2
        assert result["retry_rate"] == pytest.approx(66.67, rel=0.01)

        # 5 total attempts, 2 successful, 2 retries, 1 successful retry
        assert result["total_attempts"] == 5
        assert result["successful_attempts"] == 2
        assert result["retry_attempts"] == 2
        assert result["retry_success_rate"] == 50.0

        # Rejection reasons
        assert result["rejection_reasons"]["30% over target"] == 1
        assert result["rejection_reasons"]["20% under target"] == 1
        assert result["rejection_reasons"]["API timeout"] == 1

    def test_retry_analysis_no_retries(self) -> None:
        """Test retry analysis with no retries."""
        telemetry_data = {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "nodes": [
                        {
                            "node_id": "summary-1",
                            "node_type": "summary",
                            "level": 1,
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "status": "accepted",
                                }
                            ],
                        }
                    ]
                }
            },
        }

        result = analyze_retry_patterns(telemetry_data)

        assert result["retry_rate"] == 0.0
        assert result["retry_attempts"] == 0
        assert result["nodes_with_retries"] == 0


class TestFullMetricsComputation:
    """Test computing full IndexingMetrics from telemetry."""

    @pytest.fixture
    def config(self) -> RagZoomConfig:
        """Create test config."""
        return RagZoomConfig(
            openai_api_key="test-key",
            embedding_cost_per_1k=0.0001,
            summary_input_cost_per_1k=0.0025,
            summary_output_cost_per_1k=0.01,
        )

    @pytest.fixture
    def full_telemetry(self) -> dict:
        """Create comprehensive telemetry data."""
        return {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "nodes": [
                        {
                            "node_id": "leaf-1",
                            "node_type": "leaf",
                            "level": 0,
                            "span": [0, 100],
                            "created_at": 1234567890.0,
                            "embedding": {
                                "text_tokens": 90,
                                "batch_size": 2,
                                "batch_position": 0,
                                "model": "text-embedding-3-small",
                                "timestamp": 1234567891.0,
                            },
                        },
                        {
                            "node_id": "leaf-2",
                            "node_type": "leaf",
                            "level": 0,
                            "span": [100, 200],
                            "created_at": 1234567890.5,
                            "embedding": {
                                "text_tokens": 95,
                                "batch_size": 2,
                                "batch_position": 1,
                                "model": "text-embedding-3-small",
                                "timestamp": 1234567891.0,  # Same batch
                            },
                        },
                        {
                            "node_id": "summary-1",
                            "node_type": "summary",
                            "level": 1,
                            "span": [0, 200],
                            "created_at": 1234567892.0,
                            "embedding": {
                                "text_tokens": 85,
                                "batch_size": 1,
                                "batch_position": 0,
                                "model": "text-embedding-3-small",
                                "timestamp": 1234567893.0,
                            },
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "target_tokens": 100,
                                    "input_text_tokens": 185,  # Combined from children
                                    "prompt_tokens": 250,
                                    "completion_tokens": 90,
                                    "actual_tokens": 85,
                                    "status": "accepted",
                                    "model": "gpt-4o-mini",
                                    "timestamp": 1234567892.5,
                                }
                            ],
                        },
                    ]
                }
            },
        }

    def test_compute_full_metrics_from_telemetry(
        self, config: RagZoomConfig, full_telemetry: dict
    ) -> None:
        """Test computing full metrics from telemetry."""
        metrics = compute_metrics_from_telemetry(full_telemetry, config)

        # Check basic counts
        assert metrics.chunks_created == 2  # 2 leaf nodes
        assert metrics.embedding_api_calls == 2  # 2 unique batches
        assert metrics.summary_api_calls == 1

        # Check token totals
        assert metrics.total_embedding_tokens == 270  # 90 + 95 + 85
        assert metrics.total_summary_prompt_tokens == 250
        assert metrics.total_summary_completion_tokens == 90

        # Check timing
        assert metrics.start_time == 1234567890.0
        assert metrics.end_time == 1234567892.0

        # Check amplification metrics
        assert len(metrics.cost_amplifications) == 1
        assert len(metrics.input_amplifications) == 1
        assert len(metrics.output_amplifications) == 1

        # Check summary stats (bucketed by config.leaf_tokens)
        assert config.leaf_tokens in metrics.summary_stats
        assert metrics.summary_stats[config.leaf_tokens].count == 1

        # Check batch sizes
        assert len(metrics.embedding_batch_sizes) == 2
        assert 2 in metrics.embedding_batch_sizes
        assert 1 in metrics.embedding_batch_sizes

    def test_amplification_includes_retry_attempts(self, config: RagZoomConfig) -> None:
        """Test that amplification metrics include ALL attempts, not just accepted ones.

        This test demonstrates the bug where retries make amplification appear lower
        because only the final accepted attempt is counted.
        """
        telemetry = {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "metadata": {"source_document_tokens": 100},
                    "nodes": [
                        {
                            "node_id": "summary-1",
                            "node_type": "summary",
                            "level": 1,
                            "span": [0, 100],
                            "created_at": 1234567890.0,
                            "summary_attempts": [
                                {
                                    "is_retry": False,
                                    "target_tokens": 50,
                                    "input_text_tokens": 100,  # Text being summarized
                                    "prompt_tokens": 150,  # First attempt
                                    "completion_tokens": 120,  # Too long
                                    "actual_tokens": 120,
                                    "status": "rejected_over",
                                    "model": "gpt-4o-mini",
                                    "timestamp": 1234567891.0,
                                },
                                {
                                    "is_retry": True,
                                    "target_tokens": 50,
                                    "input_text_tokens": 100,
                                    "prompt_tokens": 160,  # Second attempt
                                    "completion_tokens": 100,  # Still too long
                                    "actual_tokens": 100,
                                    "status": "rejected_over",
                                    "model": "gpt-4o-mini",
                                    "timestamp": 1234567892.0,
                                },
                                {
                                    "is_retry": True,
                                    "target_tokens": 50,
                                    "input_text_tokens": 100,
                                    "prompt_tokens": 170,  # Third attempt
                                    "completion_tokens": 80,  # Finally accepted
                                    "actual_tokens": 80,
                                    "status": "accepted",
                                    "model": "gpt-4o-mini",
                                    "timestamp": 1234567893.0,
                                },
                            ],
                        },
                    ],
                }
            },
        }

        metrics = compute_metrics_from_telemetry(telemetry, config)

        # Verify token counts include ALL attempts
        assert metrics.total_summary_prompt_tokens == 150 + 160 + 170  # 480
        assert metrics.total_summary_completion_tokens == 120 + 100 + 80  # 300
        assert metrics.summary_api_calls == 3

        # EXPECTED behavior: Amplification should include all attempts
        # Input amplification = total_prompt_tokens / text_being_summarized
        #                    = 480 / 100 = 4.8x
        # Output amplification = total_completion_tokens / final_summary_tokens
        #                     = 300 / 80 = 3.75x

        # ACTUAL behavior (BUG): Only counts the accepted attempt
        # Input amplification = 170 / 100 = 1.7x
        # Output amplification = 80 / 80 = 1.0x

        assert len(metrics.input_amplifications) == 1
        assert len(metrics.output_amplifications) == 1

        # These assertions will FAIL, demonstrating the bug
        assert metrics.input_amplifications[0] == pytest.approx(4.8, rel=0.01)
        assert metrics.output_amplifications[0] == pytest.approx(3.75, rel=0.01)
