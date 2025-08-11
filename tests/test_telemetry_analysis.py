"""Test telemetry analysis functions."""

import pytest

from ragzoom.config import RagZoomConfig
from ragzoom.telemetry_analysis import (
    TelemetryAnalysisError,
    analyze_retry_patterns,
    compute_batch_efficiency,
    compute_metrics_from_telemetry,
    compute_simplified_metrics,
    parse_telemetry_format,
)


class TestTelemetryFormatParsing:
    """Test telemetry format parsing."""

    def test_parse_valid_telemetry_format(self) -> None:
        """Test parsing v1.0 telemetry format migrates to v3.0."""
        telemetry_data = {
            "format_version": "1.0",
            "documents": {
                "test_doc": {
                    "metadata": {
                        "source_document_tokens": 5000,
                        "chunk_size": 200,
                        "indexed_at": 1234567890.0,
                    },
                    "nodes": [
                        {
                            "node_id": "node-1",
                            "node_type": "leaf",
                            "level": 0,
                            "span": [0, 100],
                            "created_at": 1234567890.0,
                        }
                    ],
                }
            },
        }

        result = parse_telemetry_format(telemetry_data)
        # Should be migrated to v3.0
        assert result["format_version"] == "3.0"
        assert result["document_id"] == "test_doc"
        assert result["source_document_tokens"] == 5000
        assert result["chunk_size"] == 200
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["node_id"] == "node-1"

    def test_parse_v2_telemetry_format(self) -> None:
        """Test parsing v2.0 telemetry format migrates to v3.0."""
        telemetry_data = {
            "format_version": "2.0",
            "documents": {
                "test_doc": {
                    "metadata": {
                        "source_document_tokens": 5000,
                        "chunk_size": 200,
                        "indexed_at": 1234567890.0,
                    },
                    "nodes": [
                        {
                            "node_id": "node-1",
                            "height": 0,  # v2.0 uses height instead of level
                            "created_at": 1234567890.0,
                            # v2.0 doesn't have node_type or span fields
                        }
                    ],
                }
            },
        }

        result = parse_telemetry_format(telemetry_data)
        # Should be migrated to v3.0
        assert result["format_version"] == "3.0"
        assert result["document_id"] == "test_doc"
        assert result["source_document_tokens"] == 5000
        assert result["chunk_size"] == 200
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["node_id"] == "node-1"

    def test_parse_v3_telemetry_format(self) -> None:
        """Test parsing v3.0 telemetry format (already flat)."""
        telemetry_data = {
            "format_version": "3.0",
            "document_id": "test_doc",
            "source_document_tokens": 5000,
            "chunk_size": 200,
            "indexed_at": 1234567890.0,
            "models": {
                "summary": "gpt-4o-mini",
                "embedding": "text-embedding-3-small",
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
        assert result["chunk_size"] == 200
        assert result["models"]["summary"] == "gpt-4o-mini"
        assert result["models"]["embedding"] == "text-embedding-3-small"
        assert len(result["nodes"]) == 1
        assert result["nodes"][0]["node_id"] == "node-1"

    def test_parse_cli_wrapper_format_migration(self) -> None:
        """Test parsing old CLI wrapper format migrates to v3.0."""
        # This is the format the CLI used to output before v3.0
        cli_wrapper_data = {
            "config": {
                "leaf_tokens": 200,
                "budget_tokens": 8000,
                "summary_model": "gpt-4o-mini",
                "embedding_model": "text-embedding-3-small",
            },
            "document": {
                "document_id": "example.pdf",
                "file_path": "/path/to/example.pdf",
            },
            "telemetry": {
                "format_version": "2.0",
                "documents": {
                    "example.pdf": {
                        "metadata": {
                            "source_document_tokens": 7500,
                            "chunk_size": 200,
                            "indexed_at": 1234567890.0,
                        },
                        "nodes": [
                            {
                                "node_id": "node-1",
                                "height": 0,
                                "created_at": 1234567890.0,
                                "embedding": {
                                    "text_tokens": 150,
                                    "batch_size": 10,
                                    "batch_position": 3,
                                    "model": "text-embedding-3-small",
                                    "start_time": 1234567890.0,
                                    "end_time": 1234567890.5,
                                },
                            },
                            {
                                "node_id": "node-2",
                                "height": 1,
                                "created_at": 1234567890.2,
                                "summary_attempts": [
                                    {
                                        "target_tokens": 100,
                                        "prompt_tokens": 320,
                                        "completion_tokens": 95,
                                        "actual_tokens": 95,
                                        "status": "accepted",
                                        "model": "gpt-4o-mini",
                                        "start_time": 1234567890.1,
                                        "end_time": 1234567890.3,
                                    }
                                ],
                            },
                        ],
                    }
                },
            },
        }

        result = parse_telemetry_format(cli_wrapper_data)

        # Should be migrated to v3.0
        assert result["format_version"] == "3.0"
        assert result["document_id"] == "example.pdf"
        assert result["source_document_tokens"] == 7500
        assert result["chunk_size"] == 200
        assert result["indexed_at"] == 1234567890.0

        # Models should be extracted from config
        assert result["models"]["summary"] == "gpt-4o-mini"
        assert result["models"]["embedding"] == "text-embedding-3-small"

        # Nodes should be flattened
        assert len(result["nodes"]) == 2
        assert result["nodes"][0]["node_id"] == "node-1"
        assert result["nodes"][1]["node_id"] == "node-2"

        # Node data should be preserved
        assert result["nodes"][0]["embedding"]["model"] == "text-embedding-3-small"
        assert result["nodes"][1]["summary_attempts"][0]["model"] == "gpt-4o-mini"

        # Should NOT have nested documents or config/document wrappers
        assert "documents" not in result
        assert "config" not in result
        assert "document" not in result
        assert "telemetry" not in result

    def test_parse_missing_format_version(self) -> None:
        """Test parsing telemetry without format version."""
        telemetry_data = {"documents": {}}

        with pytest.raises(TelemetryAnalysisError, match="Missing format_version"):
            parse_telemetry_format(telemetry_data)

    def test_parse_unsupported_format_version(self) -> None:
        """Test parsing telemetry with unsupported format version."""
        telemetry_data = {"format_version": "4.0", "documents": {}}

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
                                    "prompt_tokens": 250,
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
                                    "prompt_tokens": 300,
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

    def test_compute_simplified_metrics(
        self, config: RagZoomConfig, sample_telemetry: dict
    ) -> None:
        """Test computing simplified metrics from telemetry."""
        result = compute_simplified_metrics(sample_telemetry, config)

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

    def test_simplified_metrics_empty_data(self, config: RagZoomConfig) -> None:
        """Test simplified metrics with empty telemetry."""
        empty_telemetry = {"format_version": "1.0", "documents": {}}

        result = compute_simplified_metrics(empty_telemetry, config)

        # Should return empty metrics
        assert result.metrics_by_chunk_size == {}

    def test_simplified_metrics_only_leaf_nodes(self, config: RagZoomConfig) -> None:
        """Test simplified metrics with only leaf nodes (no summaries)."""
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

        result = compute_simplified_metrics(leaf_only_telemetry, config)

        # Should return empty metrics (no summary attempts)
        assert result.metrics_by_chunk_size == {}

    def test_simplified_metrics_cost_calculations(
        self, config: RagZoomConfig, sample_telemetry: dict
    ) -> None:
        """Test that cost calculations in simplified metrics are correct."""
        result = compute_simplified_metrics(sample_telemetry, config)

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
            # Cost = (550 / 1000 * 0.0025) + (200 / 1000 * 0.01) = 0.001375 + 0.002 = 0.003375
            expected_total_cost = 0.003375
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
        # Efficiency: percentage of embeddings that were batched
        # Batch size 3 → 2 embeddings batched, Batch size 2 → 1 embedding batched
        # Total batched: 3, Total embeddings: 3 → 100% efficiency
        assert result["batch_utilization"] == pytest.approx(100.0, rel=0.01)

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

    def test_analyze_retry_patterns_v2_format(self) -> None:
        """Test retry analysis with v2.0 format (no is_retry field)."""
        telemetry_data = {
            "format_version": "2.0",
            "documents": {
                "test_doc": {
                    "metadata": {},
                    "nodes": [
                        {
                            "node_id": "node-1",
                            "height": 1,  # Non-leaf node
                            "summary_attempts": [
                                {
                                    "status": "rejected_over",
                                    "rejection_reason": "25% over target",
                                    "start_time": 1000.0,
                                    "end_time": 1002.0,
                                },
                                {
                                    "status": "rejected_under",
                                    "rejection_reason": "15% under target",
                                    "start_time": 1002.0,
                                    "end_time": 1004.0,
                                },
                                {
                                    "status": "accepted",
                                    "start_time": 1004.0,
                                    "end_time": 1006.0,
                                },
                            ],
                        },
                        {
                            "node_id": "node-2",
                            "height": 1,
                            "summary_attempts": [
                                {
                                    "status": "accepted",
                                    "start_time": 1000.0,
                                    "end_time": 1003.0,
                                }
                            ],
                        },
                    ],
                }
            },
        }

        result = analyze_retry_patterns(telemetry_data)

        # 2 nodes, 1 needed retries
        assert result["total_nodes_with_summaries"] == 2
        assert result["nodes_with_retries"] == 1
        assert result["retry_rate"] == 50.0

        # 4 total attempts (3 + 1), 2 retries, 1 successful retry
        assert result["total_attempts"] == 4
        assert result["successful_attempts"] == 2
        assert result["retry_attempts"] == 2
        assert (
            result["retry_success_rate"] == 50.0
        )  # Only the last retry (index 2) is accepted

        # Retry distribution
        assert result["retry_distribution"]["0"] == 1  # node-2
        assert result["retry_distribution"]["1"] == 0
        assert result["retry_distribution"]["2"] == 1  # node-1
        assert result["avg_retries_per_node"] == 1.0  # 2 retries / 2 nodes
        assert result["max_retries"] == 2

        # Timing metrics
        assert result["retry_time_seconds"] == 4.0  # 2s + 2s for the two retries
        assert result["avg_time_per_retry"] == 2.0  # 4s / 2 retries
        assert (
            result["time_wasted_on_rejections"] == 2.0
        )  # Only first retry was rejected


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

    def test_metrics_include_retry_attempts(self, config: RagZoomConfig) -> None:
        """Test that metrics include ALL attempts, not just accepted ones."""
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

        # Verify that ALL attempts are counted in the totals
        # We should have 3 summary attempts total


class TestBackwardCompatibility:
    """Test backward compatibility with v1.0 telemetry format."""

    @pytest.fixture
    def config(self) -> RagZoomConfig:
        """Create test config."""
        return RagZoomConfig(
            openai_api_key="test-key",
            embedding_cost_per_1k=0.0001,
            summary_input_cost_per_1k=0.0025,
            summary_output_cost_per_1k=0.01,
        )

    def test_v1_telemetry_with_v2_analysis(self, config: RagZoomConfig) -> None:
        """Test that v1.0 telemetry can be analyzed with v2.0 code."""
        # v1.0 telemetry with all legacy fields
        v1_telemetry = {
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
                                "timestamp": 1234567891.0,  # v1.0 single timestamp
                            },
                        },
                        {
                            "node_id": "summary-1",
                            "node_type": "summary",
                            "level": 1,
                            "span": [0, 200],
                            "created_at": 1234567892.0,
                            "summary_attempts": [
                                {
                                    "is_retry": False,  # v1.0 explicit is_retry
                                    "target_tokens": 100,
                                    "prompt_tokens": 250,
                                    "completion_tokens": 90,
                                    "actual_tokens": 85,
                                    "status": "accepted",
                                    "model": "gpt-4o-mini",
                                    "timestamp": 1234567892.5,  # v1.0 single timestamp
                                }
                            ],
                        },
                    ]
                }
            },
        }

        # All analysis functions should work with v1.0 data
        simplified = compute_simplified_metrics(v1_telemetry, config)
        assert isinstance(simplified.metrics_by_chunk_size, dict)
        # Should have metrics for chunk size 100 (the target_tokens value)

        batch_efficiency = compute_batch_efficiency(v1_telemetry)
        assert batch_efficiency["total_embeddings"] == 1

        retry_patterns = analyze_retry_patterns(v1_telemetry)
        assert retry_patterns["total_nodes_with_summaries"] == 1

        metrics = compute_metrics_from_telemetry(v1_telemetry, config)
        assert metrics.chunks_created == 1  # Should identify leaf nodes by node_type
