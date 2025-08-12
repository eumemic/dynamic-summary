"""Test telemetry analysis functions."""

import pytest

from ragzoom.telemetry_analysis import (
    TelemetryAnalysisError,
    analyze_retry_patterns,
    compute_batch_efficiency,
    compute_metrics_from_telemetry,
    compute_simplified_metrics,
    detect_verbatim_concatenations,
    parse_telemetry_format,
)


class TestTelemetryFormatParsing:
    """Test telemetry format parsing."""

    def test_parse_v3_telemetry_format(self) -> None:
        """Test parsing v3.0 telemetry format (already flat)."""
        telemetry_data = {
            "format_version": "3.0",
            "document_id": "test_doc",
            "source_document_tokens": 5000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 200,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "node-1",
                    "height": 0,
                    "created_at": 1234567890.0,
                }
            ],
        }

        result = parse_telemetry_format(telemetry_data)
        # Should remain as v3.0
        assert result["format_version"] == "3.0"
        assert result["document_id"] == "test_doc"
        assert result["source_document_tokens"] == 5000
        assert result["config"]["target_chunk_tokens"] == 200
        assert result["config"]["summary_model"] == "gpt-4o-mini"
        assert result["config"]["embedding_model"] == "text-embedding-3-small"
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["node_id"] == "node-1"

    def test_parse_v3_1_telemetry_format(self) -> None:
        """Test parsing v3.1 telemetry format."""
        telemetry_data = {
            "format_version": "3.1",
            "document_id": "test_doc",
            "source_document_tokens": 5000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 200,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "node-1",
                    "height": 0,
                    "created_at": 1234567890.0,
                }
            ],
        }

        result = parse_telemetry_format(telemetry_data)
        assert result["format_version"] == "3.1"
        assert result["document_id"] == "test_doc"
        assert result["source_document_tokens"] == 5000
        assert result["config"]["target_chunk_tokens"] == 200
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["node_id"] == "node-1"

    def test_parse_v4_1_telemetry_format(self) -> None:
        """Test parsing v4.1 telemetry format."""
        telemetry_data = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 5000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 200,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "node-1",
                    "height": 0,
                    "created_at": 1234567890.0,
                }
            ],
        }

        result = parse_telemetry_format(telemetry_data)
        assert result["format_version"] == "4.1"
        assert result["document_id"] == "test_doc"
        assert result["source_document_tokens"] == 5000
        assert result["config"]["target_chunk_tokens"] == 200
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["node_id"] == "node-1"

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


class TestTargetFitMetrics:
    """Test target fit metrics computation."""

    def test_compute_median_error_new_format(self) -> None:
        """Test median error computation with new format (no status field)."""
        nodes = [
            {
                "node_id": "node-1",
                "height": 1,
                "created_at": 1234567890.0,
                "summary_attempts": [
                    {
                        "target_tokens": 100,
                        "actual_tokens": 95,
                        "prompt_tokens": 300,
                        "completion_tokens": 95,
                        "model": "gpt-4o-mini",
                        "start_time": 1234567890.0,
                        "end_time": 1234567891.0,
                    }
                ],
                "accepted_attempt": 0,
            },
            {
                "node_id": "node-2",
                "height": 1,
                "created_at": 1234567891.0,
                "summary_attempts": [
                    {
                        "target_tokens": 100,
                        "actual_tokens": 110,
                        "prompt_tokens": 300,
                        "completion_tokens": 110,
                        "model": "gpt-4o-mini",
                        "start_time": 1234567891.0,
                        "end_time": 1234567892.0,
                    },
                    {
                        "target_tokens": 100,
                        "actual_tokens": 105,
                        "prompt_tokens": 350,
                        "completion_tokens": 105,
                        "model": "gpt-4o-mini",
                        "start_time": 1234567892.0,
                        "end_time": 1234567893.0,
                    },
                ],
                "accepted_attempt": 1,  # Use second attempt
            },
            {
                "node_id": "node-3",
                "height": 1,
                "created_at": 1234567893.0,
                "summary_attempts": [
                    {
                        "target_tokens": 100,
                        "actual_tokens": 98,
                        "prompt_tokens": 300,
                        "completion_tokens": 98,
                        "model": "gpt-4o-mini",
                        "start_time": 1234567893.0,
                        "end_time": 1234567894.0,
                    }
                ],
                # No accepted_attempt field - should use last attempt
            },
        ]

        from ragzoom.telemetry_analysis import compute_target_fit_metrics

        result = compute_target_fit_metrics(nodes, target_size=100)

        # Errors should be: -5 (95-100), +5 (105-100), -2 (98-100)
        # Median of [-5, -2, 5] = -2
        assert result["median_error"] == -2.0
        assert result["percent_within_10"] == 100.0  # All within ±10

    def test_compute_median_error_backward_compat(self) -> None:
        """Test median error with format without accepted_attempt field."""
        nodes = [
            {
                "node_id": "node-1",
                "height": 1,
                "created_at": 1234567890.0,
                "summary_attempts": [
                    {
                        "target_tokens": 100,
                        "actual_tokens": 120,
                        "prompt_tokens": 300,
                        "completion_tokens": 120,
                        "model": "gpt-4o-mini",
                        "start_time": 1234567890.0,
                        "end_time": 1234567891.0,
                    },
                    {
                        "target_tokens": 100,
                        "actual_tokens": 90,
                        "prompt_tokens": 350,
                        "completion_tokens": 90,
                        "model": "gpt-4o-mini",
                        "start_time": 1234567891.0,
                        "end_time": 1234567892.0,
                    },
                ],
            },
        ]

        from ragzoom.telemetry_analysis import compute_target_fit_metrics

        result = compute_target_fit_metrics(nodes, target_size=100)

        # Should use the last attempt (90 tokens)
        # Error should be: -10 (90-100)
        assert result["median_error"] == -10.0
        assert result["percent_within_10"] == 100.0


class TestSimplifiedMetrics:
    """Test simplified metrics computation."""

    # Config fixture removed - telemetry functions no longer need config

    @pytest.fixture
    def sample_telemetry(self) -> dict:
        """Create sample telemetry data with summary attempts."""
        return {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 1000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "summary-1",
                    "height": 1,
                    "created_at": 1234567890.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 250,
                            "completion_tokens": 90,
                            "actual_tokens": 90,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567890.5,
                            "end_time": 1234567891.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
                {
                    "node_id": "summary-2",
                    "height": 1,
                    "created_at": 1234567892.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 300,
                            "completion_tokens": 110,
                            "actual_tokens": 100,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.5,
                            "end_time": 1234567893.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
            ],
        }

    def test_compute_simplified_metrics(self, sample_telemetry: dict) -> None:
        """Test computing simplified metrics from telemetry."""
        result = compute_simplified_metrics(sample_telemetry)

        # Should have metrics organized by chunk size
        assert hasattr(result, "metrics_by_chunk_size")
        assert isinstance(result.metrics_by_chunk_size, dict)

        # For each chunk size with data
        for chunk_size, metrics in result.metrics_by_chunk_size.items():
            # Should have all metric categories
            assert "target_fit" in metrics
            assert "retries" in metrics
            assert "latency" in metrics
            assert "cost" in metrics
            assert "dispersion" in metrics

            # Target-fit metrics
            assert "median_error" in metrics["target_fit"]
            assert "p95_error" in metrics["target_fit"]
            assert "percent_within_10" in metrics["target_fit"]

            # Retry metrics
            assert "retry_rate" in metrics["retries"]
            assert "max_retries" in metrics["retries"]

    def test_simplified_metrics_empty_data(self) -> None:
        """Test simplified metrics with empty telemetry."""
        empty_telemetry = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 0,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [],
        }

        result = compute_simplified_metrics(empty_telemetry)

        # Should return empty metrics
        assert result.metrics_by_chunk_size == {}

    def test_simplified_metrics_only_leaf_nodes(self) -> None:
        """Test simplified metrics with only leaf nodes (no summaries)."""
        leaf_only_telemetry = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 1000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "leaf-1",
                    "height": 0,
                    "created_at": 1234567890.0,
                }
            ],
        }

        result = compute_simplified_metrics(leaf_only_telemetry)

        # Should return empty metrics (no summary attempts)
        assert result.metrics_by_chunk_size == {}

    def test_simplified_metrics_cost_calculations(self, sample_telemetry: dict) -> None:
        """Test that cost calculations in simplified metrics are correct."""
        result = compute_simplified_metrics(sample_telemetry)

        # Verify cost calculations for the chunk size
        for chunk_size, metrics in result.metrics_by_chunk_size.items():
            cost_metrics = metrics["cost"]

            # Check that cost metrics exist and are reasonable
            assert "usd_per_node" in cost_metrics
            assert "total_prompt_tokens" in cost_metrics
            assert "total_completion_tokens" in cost_metrics
            assert cost_metrics["usd_per_node"] > 0
            assert cost_metrics["total_prompt_tokens"] > 0

            # Verify cost calculation is correct
            # Based on sample data: 2 nodes with 250 + 300 = 550 prompt tokens, 90 + 110 = 200 completion tokens
            # Using gpt-4o-mini pricing from pricing.json: $0.00015/1K input, $0.0006/1K output
            # Cost = (550 / 1000 * 0.00015) + (200 / 1000 * 0.0006) = 0.0000825 + 0.00012 = 0.0002025
            expected_total_cost = 0.0002025
            # USD per node (2 nodes)
            expected_usd_per_node = expected_total_cost / 2
            assert cost_metrics["usd_per_node"] == pytest.approx(
                expected_usd_per_node, rel=0.01
            )

            # Verify token counts
            assert cost_metrics["total_prompt_tokens"] == 550
            assert cost_metrics["total_completion_tokens"] == 200


class TestBatchEfficiency:
    """Test batch efficiency analysis."""

    def test_compute_batch_efficiency(self) -> None:
        """Test computing batch efficiency from telemetry."""
        telemetry_data = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 1000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "node-1",
                    "height": 0,
                    "created_at": 1234567890.0,
                    "embedding": {
                        "text_tokens": 50,
                        "batch_size": 3,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567890.0,
                        "end_time": 1234567890.5,
                    },
                },
                {
                    "node_id": "node-2",
                    "height": 0,
                    "created_at": 1234567890.0,
                    "embedding": {
                        "text_tokens": 45,
                        "batch_size": 3,
                        "batch_position": 1,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567890.0,
                        "end_time": 1234567890.5,
                    },
                },
                {
                    "node_id": "node-3",
                    "height": 0,
                    "created_at": 1234567891.0,
                    "embedding": {
                        "text_tokens": 55,
                        "batch_size": 2,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567891.0,
                        "end_time": 1234567891.5,
                    },
                },
            ],
        }

        result = compute_batch_efficiency(telemetry_data)

        # Should have 2 unique batches (sizes 3 and 2)
        assert result["total_batches"] == 2
        assert result["total_embeddings"] == 3
        assert result["batch_sizes"] == [3, 2]
        assert result["avg_batch_size"] == 2.5
        # Efficiency: percentage of embeddings that were batched
        # Batch size 3 → 2 embeddings batched, Batch size 2 → 1 embedding batched
        # Total batched: 3, Total embeddings: 3 → 100% efficiency
        assert result["batch_utilization"] == pytest.approx(100.0, rel=0.01)

    def test_batch_efficiency_empty_data(self) -> None:
        """Test batch efficiency with empty telemetry."""
        empty_telemetry = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 0,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [],
        }

        result = compute_batch_efficiency(empty_telemetry)

        assert result["total_batches"] == 0
        assert result["total_embeddings"] == 0
        assert result["batch_sizes"] == []
        assert result["avg_batch_size"] == 0.0
        assert result["batch_utilization"] == 0.0


class TestRetryAnalysis:
    """Test retry pattern analysis."""

    def test_successful_attempts_equals_node_count_new_format(self) -> None:
        """Test that successful attempts equals number of summary nodes (new format)."""
        # Create telemetry with 3 summary nodes, some with multiple attempts
        telemetry_data = {
            "format_version": "3.0",
            "document_id": "test",
            "source_document_tokens": 1000,
            "chunk_size": 100,
            "indexed_at": 1234567890.0,
            "models": {
                "summary": "gpt-4o-mini",
                "embedding": "text-embedding-3-small",
            },
            "nodes": [
                # Leaf node - should be skipped
                {
                    "node_id": "leaf-1",
                    "height": 0,
                    "created_at": 1234567890.0,
                },
                # Summary node 1 - single attempt
                {
                    "node_id": "summary-1",
                    "height": 1,
                    "created_at": 1234567891.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 300,
                            "completion_tokens": 105,
                            "actual_tokens": 105,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567891.0,
                            "end_time": 1234567892.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
                # Summary node 2 - multiple attempts
                {
                    "node_id": "summary-2",
                    "height": 1,
                    "created_at": 1234567892.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 300,
                            "completion_tokens": 130,
                            "actual_tokens": 130,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.0,
                            "end_time": 1234567893.0,
                        },
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 350,
                            "completion_tokens": 95,
                            "actual_tokens": 95,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567893.0,
                            "end_time": 1234567894.0,
                        },
                    ],
                    "accepted_attempt": 1,
                },
                # Summary node 3 - three attempts
                {
                    "node_id": "summary-3",
                    "height": 1,
                    "created_at": 1234567894.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 300,
                            "completion_tokens": 140,
                            "actual_tokens": 140,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567894.0,
                            "end_time": 1234567895.0,
                        },
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 350,
                            "completion_tokens": 85,
                            "actual_tokens": 85,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567895.0,
                            "end_time": 1234567896.0,
                        },
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 400,
                            "completion_tokens": 102,
                            "actual_tokens": 102,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567896.0,
                            "end_time": 1234567897.0,
                        },
                    ],
                    "accepted_attempt": 2,
                },
            ],
        }

        result = analyze_retry_patterns(telemetry_data)

        # Should have exactly 3 successful attempts (one per summary node)
        assert result["successful_attempts"] == 3
        assert result["total_nodes_with_summaries"] == 3
        # Total attempts should be 6 (1 + 2 + 3)
        assert result["total_attempts"] == 6

    def test_successful_attempts_backward_compat_no_accepted_field(self) -> None:
        """Test backward compatibility without accepted_attempt field."""
        # Format without accepted_attempt field (should use last attempt)
        telemetry_data = {
            "format_version": "3.0",
            "document_id": "test",
            "source_document_tokens": 1000,
            "chunk_size": 100,
            "indexed_at": 1234567890.0,
            "models": {
                "summary": "gpt-4o-mini",
                "embedding": "text-embedding-3-small",
            },
            "nodes": [
                # Summary node with multiple attempts, no accepted_attempt field
                {
                    "node_id": "summary-1",
                    "height": 1,
                    "created_at": 1234567891.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 300,
                            "completion_tokens": 130,
                            "actual_tokens": 130,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567891.0,
                            "end_time": 1234567892.0,
                        },
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 350,
                            "completion_tokens": 95,
                            "actual_tokens": 95,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.0,
                            "end_time": 1234567893.0,
                        },
                    ],
                    # No accepted_attempt field (should use last attempt)
                },
            ],
        }

        result = analyze_retry_patterns(telemetry_data)

        # Should count exactly 1 successful attempt (last one)
        assert result["successful_attempts"] == 1
        assert result["total_nodes_with_summaries"] == 1

    def test_analyze_retry_patterns(self) -> None:
        """Test analyzing retry patterns from telemetry."""
        telemetry_data = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 1000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "summary-1",
                    "height": 1,
                    "created_at": 1234567890.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 250,
                            "completion_tokens": 130,
                            "actual_tokens": 130,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567890.0,
                            "end_time": 1234567891.0,
                        },
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 250,
                            "completion_tokens": 95,
                            "actual_tokens": 95,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567891.0,
                            "end_time": 1234567892.0,
                        },
                    ],
                    "accepted_attempt": 1,
                },
                {
                    "node_id": "summary-2",
                    "height": 1,
                    "created_at": 1234567892.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 300,
                            "completion_tokens": 100,
                            "actual_tokens": 100,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.0,
                            "end_time": 1234567893.0,
                        }
                    ],
                    "accepted_attempt": 0,
                },
                {
                    "node_id": "summary-3",
                    "height": 1,
                    "created_at": 1234567893.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 250,
                            "completion_tokens": 80,
                            "actual_tokens": 80,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567893.0,
                            "end_time": 1234567894.0,
                        },
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 250,
                            "completion_tokens": 95,
                            "actual_tokens": 95,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567894.0,
                            "end_time": 1234567894.5,
                        },
                    ],
                    "accepted_attempt": 1,
                },
            ],
        }

        result = analyze_retry_patterns(telemetry_data)

        # 3 summary nodes total, 2 needed retries
        assert result["total_nodes_with_summaries"] == 3
        assert result["nodes_with_retries"] == 2
        assert result["retry_rate"] == pytest.approx(66.67, rel=0.01)

        # 5 total attempts, 3 successful (one per node), 2 retries, 2 successful retries
        assert result["total_attempts"] == 5
        assert result["successful_attempts"] == 3  # One per node, regardless of status
        assert result["retry_attempts"] == 2
        # Both retries are the final attempts for their nodes, so both are "successful"
        assert result["retry_success_rate"] == 100.0

        # Rejection reasons - we no longer track these since we removed status field
        # Old telemetry with status fields is just for backward compat, not analysis
        assert result["rejection_reasons"] == {}

        # New metrics should be present
        assert "retry_distribution" in result
        assert result["retry_distribution"]["0"] == 1  # summary-2 has 0 retries
        assert (
            result["retry_distribution"]["1"] == 2
        )  # summary-1 and summary-3 have 1 retry each
        assert result["avg_retries_per_node"] == pytest.approx(0.67, rel=0.01)  # 2/3
        assert result["max_retries"] == 1

    def test_retry_analysis_no_retries(self) -> None:
        """Test retry analysis with no retries."""
        telemetry_data = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 1000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "summary-1",
                    "height": 1,
                    "created_at": 1234567890.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 250,
                            "completion_tokens": 95,
                            "actual_tokens": 95,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567890.0,
                            "end_time": 1234567891.0,
                        }
                    ],
                    "accepted_attempt": 0,
                }
            ],
        }

        result = analyze_retry_patterns(telemetry_data)

        assert result["retry_rate"] == 0.0
        assert result["retry_attempts"] == 0
        assert result["nodes_with_retries"] == 0


class TestFullMetricsComputation:
    """Test computing full IndexingMetrics from telemetry."""

    # Config fixture removed - telemetry functions no longer need config

    @pytest.fixture
    def full_telemetry(self) -> dict:
        """Create comprehensive telemetry data."""
        return {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 2000,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 100,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "leaf-1",
                    "height": 0,
                    "created_at": 1234567890.0,
                    "embedding": {
                        "text_tokens": 90,
                        "batch_size": 2,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567891.0,
                        "end_time": 1234567891.5,
                    },
                },
                {
                    "node_id": "leaf-2",
                    "height": 0,
                    "created_at": 1234567890.5,
                    "embedding": {
                        "text_tokens": 95,
                        "batch_size": 2,
                        "batch_position": 1,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567891.0,
                        "end_time": 1234567891.5,
                    },
                },
                {
                    "node_id": "summary-1",
                    "height": 1,
                    "created_at": 1234567892.0,
                    "embedding": {
                        "text_tokens": 85,
                        "batch_size": 1,
                        "batch_position": 0,
                        "model": "text-embedding-3-small",
                        "start_time": 1234567893.0,
                        "end_time": 1234567893.5,
                    },
                    "summary_attempts": [
                        {
                            "target_tokens": 100,
                            "prompt_tokens": 250,
                            "completion_tokens": 90,
                            "actual_tokens": 85,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.0,
                            "end_time": 1234567892.5,
                        }
                    ],
                    "accepted_attempt": 0,
                },
            ],
        }

    def test_compute_full_metrics_from_telemetry(self, full_telemetry: dict) -> None:
        """Test computing full metrics from telemetry."""
        metrics = compute_metrics_from_telemetry(full_telemetry)

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

        # Check basic metrics
        assert metrics.chunks_created == 2  # 2 leaf nodes
        assert metrics.summary_api_calls == 1

        # Check summary stats (bucketed by actual target_tokens from telemetry)
        assert 100 in metrics.summary_stats  # The test data has target_tokens=100
        assert metrics.summary_stats[100].count == 1

        # Check batch sizes
        assert len(metrics.embedding_batch_sizes) == 2
        assert 2 in metrics.embedding_batch_sizes
        assert 1 in metrics.embedding_batch_sizes

    def test_metrics_include_retry_attempts(self) -> None:
        """Test that metrics include ALL attempts, not just accepted ones."""
        telemetry = {
            "format_version": "4.1",
            "document_id": "test_doc",
            "source_document_tokens": 100,
            "indexed_at": 1234567890.0,
            "config": {
                "target_chunk_tokens": 50,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "nodes": [
                {
                    "node_id": "summary-1",
                    "height": 1,
                    "created_at": 1234567890.0,
                    "summary_attempts": [
                        {
                            "target_tokens": 50,
                            "prompt_tokens": 150,
                            "completion_tokens": 120,
                            "actual_tokens": 120,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567891.0,
                            "end_time": 1234567891.5,
                        },
                        {
                            "target_tokens": 50,
                            "prompt_tokens": 160,
                            "completion_tokens": 100,
                            "actual_tokens": 100,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567892.0,
                            "end_time": 1234567892.5,
                        },
                        {
                            "target_tokens": 50,
                            "prompt_tokens": 170,
                            "completion_tokens": 80,
                            "actual_tokens": 80,
                            "model": "gpt-4o-mini",
                            "start_time": 1234567893.0,
                            "end_time": 1234567893.5,
                        },
                    ],
                    "accepted_attempt": 2,
                },
            ],
        }

        metrics = compute_metrics_from_telemetry(telemetry)

        # Verify token counts include ALL attempts
        assert metrics.total_summary_prompt_tokens == 150 + 160 + 170  # 480
        assert metrics.total_summary_completion_tokens == 120 + 100 + 80  # 300
        assert metrics.summary_api_calls == 3

        # Verify that ALL attempts are counted in the totals
        # We should have 3 summary attempts total


class TestVerbatimDetection:
    """Test verbatim concatenation detection."""

    def test_detect_no_verbatim(self) -> None:
        """Test detection when no verbatim issues exist."""
        nodes = [
            {
                "node_id": "node-1",
                "height": 1,
                "summary_attempts": [
                    {
                        "prompt_tokens": 400,
                        "completion_tokens": 200,  # 0.5 ratio - good compression
                    }
                ],
            },
            {
                "node_id": "node-2",
                "height": 2,
                "summary_attempts": [
                    {
                        "prompt_tokens": 600,
                        "completion_tokens": 250,  # 0.42 ratio - good compression
                    }
                ],
            },
        ]

        result = detect_verbatim_concatenations(nodes)
        assert result["total_summaries"] == 2
        assert result["verbatim_count"] == 0
        assert result["verbatim_percentage"] == 0.0
        assert len(result["worst_offenders"]) == 0

    def test_detect_verbatim_issues(self) -> None:
        """Test detection of verbatim concatenations."""
        nodes = [
            {
                "node_id": "node-1",
                "height": 1,
                "summary_attempts": [
                    {
                        "prompt_tokens": 400,
                        "completion_tokens": 398,  # 0.995 ratio - verbatim!
                    }
                ],
            },
            {
                "node_id": "node-2",
                "height": 2,
                "summary_attempts": [
                    {
                        "prompt_tokens": 600,
                        "completion_tokens": 600,  # 1.0 ratio - exact verbatim!
                    }
                ],
            },
            {
                "node_id": "node-3",
                "height": 1,
                "summary_attempts": [
                    {
                        "prompt_tokens": 300,
                        "completion_tokens": 150,  # 0.5 ratio - good
                    }
                ],
            },
        ]

        result = detect_verbatim_concatenations(nodes)
        assert result["total_summaries"] == 3
        assert result["verbatim_count"] == 2
        assert result["verbatim_percentage"] == pytest.approx(66.67, 0.1)
        assert len(result["worst_offenders"]) == 2

        # Check worst offender is the 600-token one
        worst = result["worst_offenders"][0]
        assert worst["input_tokens"] == 600
        assert worst["output_tokens"] == 600
        assert worst["ratio"] == 1.0

    def test_detect_verbatim_with_tolerance(self) -> None:
        """Test detection with custom tolerance threshold."""
        nodes = [
            {
                "node_id": "node-1",
                "height": 1,
                "summary_attempts": [
                    {
                        "prompt_tokens": 400,
                        "completion_tokens": 394,  # 0.985 ratio - within 2% default tolerance
                    }
                ],
            },
            {
                "node_id": "node-2",
                "height": 1,
                "summary_attempts": [
                    {
                        "prompt_tokens": 400,
                        "completion_tokens": 382,  # 0.955 ratio - outside 2% tolerance, within 5%
                    }
                ],
            },
        ]

        # With default 2% tolerance
        result = detect_verbatim_concatenations(nodes, tolerance=0.02)
        assert result["verbatim_count"] == 1  # Only the 0.98 ratio

        # With 5% tolerance
        result = detect_verbatim_concatenations(nodes, tolerance=0.05)
        assert result["verbatim_count"] == 2  # Both are caught

    def test_height_distribution(self) -> None:
        """Test height distribution tracking."""
        nodes = [
            {
                "node_id": f"node-{i}",
                "height": i % 3 + 1,  # Heights 1, 2, 3, 1, 2, 3...
                "summary_attempts": [
                    {
                        "prompt_tokens": 400,
                        "completion_tokens": (
                            399 if i < 4 else 200
                        ),  # First 4 are verbatim
                    }
                ],
            }
            for i in range(6)
        ]

        result = detect_verbatim_concatenations(nodes)
        assert result["verbatim_count"] == 4
        assert result["height_distribution"] == {1: 2, 2: 1, 3: 1}

    def test_accepted_attempt_handling(self) -> None:
        """Test that accepted_attempt index is respected."""
        nodes = [
            {
                "node_id": "node-1",
                "height": 1,
                "summary_attempts": [
                    {
                        "prompt_tokens": 400,
                        "completion_tokens": 400,  # Verbatim but not accepted
                    },
                    {
                        "prompt_tokens": 400,
                        "completion_tokens": 200,  # Good compression, accepted
                    },
                ],
                "accepted_attempt": 1,  # Second attempt is accepted
            },
        ]

        result = detect_verbatim_concatenations(nodes)
        assert result["verbatim_count"] == 0  # Accepted attempt is not verbatim
