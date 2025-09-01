"""Test cost calculations with cached token discounts using production code."""

from unittest.mock import MagicMock, patch

import pytest

from ragzoom.telemetry_analysis import compute_cost_metrics
from ragzoom.telemetry_types import NodeTelemetryDict


def test_compute_cost_metrics_with_cached_tokens() -> None:
    """Test that cached tokens receive appropriate discount in production code."""

    # Create test nodes with cached tokens
    nodes: list[NodeTelemetryDict] = [
        {
            "node_id": "node1",
            "height": 1,
            "created_at": 0.0,
            "summary_attempts": [
                {
                    "target_tokens": 100,
                    "prompt_tokens": 1500,
                    "cached_tokens": 1200,  # 80% cached
                    "completion_tokens": 100,
                    "actual_tokens": 100,
                    "model": "gpt-4o-mini",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
            "embedding": {
                "text_tokens": 500,
                "batch_size": 1,
                "batch_position": 0,
                "model": "text-embedding-3-small",
                "start_time": 0.0,
                "end_time": 0.1,
            },
        },
        {
            "node_id": "node2",
            "height": 1,
            "created_at": 1.0,
            "summary_attempts": [
                {
                    "target_tokens": 100,
                    "prompt_tokens": 2000,
                    "cached_tokens": 1800,  # 90% cached
                    "completion_tokens": 150,
                    "actual_tokens": 150,
                    "model": "gpt-4o-mini",
                    "start_time": 1.0,
                    "end_time": 2.0,
                }
            ],
            "embedding": {
                "text_tokens": 600,
                "batch_size": 1,
                "batch_position": 0,
                "model": "text-embedding-3-small",
                "start_time": 1.0,
                "end_time": 1.1,
            },
        },
    ]

    models = {"summary": "gpt-4o-mini", "embedding": "text-embedding-3-small"}
    source_tokens = 10000

    # Mock the model info to return a known cache discount
    with patch("ragzoom.model_info.ModelInfo") as mock_model_info_class:
        mock_model_info = MagicMock()
        mock_model_info.get_cache_discount.return_value = (
            0.5  # 50% discount (pay 50% of original)
        )
        mock_model_info_class.return_value = mock_model_info

        # Mock pricing info
        with patch("ragzoom.telemetry_analysis.get_model_pricing") as mock_pricing:
            mock_pricing.return_value = {
                "summary_input_cost_per_1k": 0.00015,  # gpt-4o-mini input
                "summary_output_cost_per_1k": 0.0006,  # gpt-4o-mini output
                "embedding_cost_per_1k": 0.00002,  # text-embedding-3-small
            }

            result = compute_cost_metrics(nodes, models, source_tokens)

    # Verify token counts
    assert result.total_prompt_tokens == 3500  # 1500 + 2000
    assert result.total_completion_tokens == 250  # 100 + 150

    # Calculate expected costs with cache discount
    # Node 1:
    #   - Embedding: 500 * 0.00002 / 1000 = 0.00001
    #   - Prompt: (300 * 0.00015 + 1200 * 0.00015 * 0.5) / 1000 = 0.000135
    #   - Completion: 100 * 0.0006 / 1000 = 0.00006
    #   - Total: 0.000205
    # Node 2:
    #   - Embedding: 600 * 0.00002 / 1000 = 0.000012
    #   - Prompt: (200 * 0.00015 + 1800 * 0.00015 * 0.5) / 1000 = 0.000165
    #   - Completion: 150 * 0.0006 / 1000 = 0.00009
    #   - Total: 0.000267
    # Total cost: 0.000472
    expected_total_cost = 0.000472
    expected_per_node = expected_total_cost / 2
    expected_per_million = (expected_total_cost / source_tokens) * 1_000_000

    assert result.usd_per_node == pytest.approx(expected_per_node, rel=1e-4)
    assert result.usd_per_million_source_tokens == pytest.approx(
        expected_per_million, rel=1e-4
    )


def test_compute_cost_metrics_without_cached_tokens() -> None:
    """Test backward compatibility when cached_tokens is not present."""

    nodes: list[NodeTelemetryDict] = [
        {
            "node_id": "node1",
            "height": 1,
            "created_at": 0.0,
            "summary_attempts": [
                {
                    "target_tokens": 100,
                    "prompt_tokens": 1500,
                    # No cached_tokens field
                    "completion_tokens": 100,
                    "actual_tokens": 100,
                    "model": "gpt-4o-mini",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
            "embedding": {
                "text_tokens": 500,
                "batch_size": 1,
                "batch_position": 0,
                "model": "text-embedding-3-small",
                "start_time": 0.0,
                "end_time": 0.1,
            },
        }
    ]

    models = {"summary": "gpt-4o-mini", "embedding": "text-embedding-3-small"}
    source_tokens = 10000

    with patch("ragzoom.telemetry_analysis.get_model_pricing") as mock_pricing:
        mock_pricing.return_value = {
            "summary_input_cost_per_1k": 0.00015,
            "summary_output_cost_per_1k": 0.0006,
            "embedding_cost_per_1k": 0.00002,
        }

        result = compute_cost_metrics(nodes, models, source_tokens)

    # Should calculate as if no tokens were cached
    # Embedding: 500 * 0.00002 / 1000 = 0.00001
    # Prompt: 1500 * 0.00015 / 1000 = 0.000225
    # Completion: 100 * 0.0006 / 1000 = 0.00006
    # Total: 0.000295
    expected_cost = 0.000295

    assert result.usd_per_node == pytest.approx(expected_cost, rel=1e-4)


def test_compute_cost_metrics_with_zero_cached_tokens() -> None:
    """Test that zero cached tokens works correctly."""

    nodes: list[NodeTelemetryDict] = [
        {
            "node_id": "node1",
            "height": 1,
            "created_at": 0.0,
            "summary_attempts": [
                {
                    "target_tokens": 100,
                    "prompt_tokens": 1500,
                    "cached_tokens": 0,  # Explicitly zero
                    "completion_tokens": 100,
                    "actual_tokens": 100,
                    "model": "gpt-4o-mini",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
            "embedding": {
                "text_tokens": 500,
                "batch_size": 1,
                "batch_position": 0,
                "model": "text-embedding-3-small",
                "start_time": 0.0,
                "end_time": 0.1,
            },
        }
    ]

    models = {"summary": "gpt-4o-mini", "embedding": "text-embedding-3-small"}
    source_tokens = 10000

    with patch("ragzoom.telemetry_analysis.get_model_pricing") as mock_pricing:
        mock_pricing.return_value = {
            "summary_input_cost_per_1k": 0.00015,
            "summary_output_cost_per_1k": 0.0006,
            "embedding_cost_per_1k": 0.00002,
        }

        result = compute_cost_metrics(nodes, models, source_tokens)

    # Same as without cached tokens
    expected_cost = 0.000295
    assert result.usd_per_node == pytest.approx(expected_cost, rel=1e-4)


def test_compute_cost_metrics_with_high_cache_rate() -> None:
    """Test cost savings with very high cache hit rate."""

    nodes: list[NodeTelemetryDict] = [
        {
            "node_id": "node1",
            "height": 1,
            "created_at": 0.0,
            "summary_attempts": [
                {
                    "target_tokens": 100,
                    "prompt_tokens": 2000,
                    "cached_tokens": 1900,  # 95% cache hit rate
                    "completion_tokens": 200,
                    "actual_tokens": 200,
                    "model": "gpt-4o",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
            "embedding": {
                "text_tokens": 1000,
                "batch_size": 1,
                "batch_position": 0,
                "model": "text-embedding-3-large",
                "start_time": 0.0,
                "end_time": 0.1,
            },
        }
    ]

    models = {"summary": "gpt-4o", "embedding": "text-embedding-3-large"}
    source_tokens = 10000

    with patch("ragzoom.model_info.ModelInfo") as mock_model_info_class:
        mock_model_info = MagicMock()
        mock_model_info.get_cache_discount.return_value = 0.5  # 50% discount
        mock_model_info_class.return_value = mock_model_info

        with patch("ragzoom.telemetry_analysis.get_model_pricing") as mock_pricing:
            mock_pricing.return_value = {
                "summary_input_cost_per_1k": 0.0025,  # gpt-4o input
                "summary_output_cost_per_1k": 0.01,  # gpt-4o output
                "embedding_cost_per_1k": 0.00013,  # text-embedding-3-large
            }

            result = compute_cost_metrics(nodes, models, source_tokens)

    # Calculate with 95% cache hit
    # Embedding: 1000 * 0.00013 / 1000 = 0.00013
    # Prompt: (100 * 0.0025 + 1900 * 0.0025 * 0.5) / 1000 = 0.002625
    # Completion: 200 * 0.01 / 1000 = 0.002
    # Total: 0.004755
    expected_cost = 0.004755

    # Now calculate without cache for comparison
    cost_without_cache = (1000 * 0.00013 + 2000 * 0.0025 + 200 * 0.01) / 1000
    assert cost_without_cache == 0.00713

    # Verify we got significant savings
    savings_percentage = (cost_without_cache - expected_cost) / cost_without_cache * 100
    assert savings_percentage > 30  # Should save > 30% with 95% cache rate

    assert result.usd_per_node == pytest.approx(expected_cost, rel=1e-4)


def test_compute_cost_metrics_with_multiple_attempts() -> None:
    """Test cost calculation when nodes have multiple summary attempts."""

    nodes: list[NodeTelemetryDict] = [
        {
            "node_id": "node1",
            "height": 1,
            "created_at": 0.0,
            "summary_attempts": [
                {
                    "target_tokens": 100,
                    "prompt_tokens": 1000,
                    "cached_tokens": 0,  # First attempt, no cache
                    "completion_tokens": 150,
                    "actual_tokens": 150,
                    "model": "gpt-4o-mini",
                    "start_time": 0.0,
                    "end_time": 1.0,
                },
                {
                    "target_tokens": 100,
                    "prompt_tokens": 1200,
                    "cached_tokens": 1000,  # Retry with cache
                    "completion_tokens": 100,
                    "actual_tokens": 100,
                    "model": "gpt-4o-mini",
                    "start_time": 1.0,
                    "end_time": 2.0,
                },
            ],
            "embedding": {
                "text_tokens": 500,
                "batch_size": 1,
                "batch_position": 0,
                "model": "text-embedding-3-small",
                "start_time": 0.0,
                "end_time": 0.1,
            },
        }
    ]

    models = {"summary": "gpt-4o-mini", "embedding": "text-embedding-3-small"}
    source_tokens = 10000

    with patch("ragzoom.model_info.ModelInfo") as mock_model_info_class:
        mock_model_info = MagicMock()
        mock_model_info.get_cache_discount.return_value = 0.5
        mock_model_info_class.return_value = mock_model_info

        with patch("ragzoom.telemetry_analysis.get_model_pricing") as mock_pricing:
            mock_pricing.return_value = {
                "summary_input_cost_per_1k": 0.00015,
                "summary_output_cost_per_1k": 0.0006,
                "embedding_cost_per_1k": 0.00002,
            }

            result = compute_cost_metrics(nodes, models, source_tokens)

    # Both attempts should be counted
    assert result.total_prompt_tokens == 2200  # 1000 + 1200
    assert result.total_completion_tokens == 250  # 150 + 100

    # Cost calculation:
    # Embedding: 500 * 0.00002 / 1000 = 0.00001
    # Attempt 1: 1000 * 0.00015 / 1000 + 150 * 0.0006 / 1000 = 0.00024
    # Attempt 2: (200 * 0.00015 + 1000 * 0.00015 * 0.5) / 1000 + 100 * 0.0006 / 1000 = 0.000165
    # Total: 0.000415
    expected_cost = 0.000415

    assert result.usd_per_node == pytest.approx(expected_cost, rel=1e-4)


def test_compute_cost_metrics_model_not_found() -> None:
    """Test graceful handling when model info is not available."""

    nodes: list[NodeTelemetryDict] = [
        {
            "node_id": "node1",
            "height": 1,
            "created_at": 0.0,
            "summary_attempts": [
                {
                    "target_tokens": 100,
                    "prompt_tokens": 1500,
                    "cached_tokens": 1200,
                    "completion_tokens": 100,
                    "actual_tokens": 100,
                    "model": "unknown-model",
                    "start_time": 0.0,
                    "end_time": 1.0,
                }
            ],
        }
    ]

    models = {"summary": "unknown-model", "embedding": "text-embedding-3-small"}
    source_tokens = 10000

    # Mock ModelInfo to raise ValueError
    with patch("ragzoom.model_info.ModelInfo") as mock_model_info_class:
        mock_model_info = MagicMock()
        mock_model_info.get_cache_discount.side_effect = ValueError("Model not found")
        mock_model_info_class.return_value = mock_model_info

        with patch("ragzoom.telemetry_analysis.get_model_pricing") as mock_pricing:
            mock_pricing.return_value = {
                "summary_input_cost_per_1k": 0.001,
                "summary_output_cost_per_1k": 0.002,
                "embedding_cost_per_1k": 0.00002,
            }

            result = compute_cost_metrics(nodes, models, source_tokens)

    # Should fall back to no discount (cache_discount = 0.0)
    # All prompt tokens charged at full price
    expected_cost = (1500 * 0.001 + 100 * 0.002) / 1000
    assert result.usd_per_node == pytest.approx(expected_cost, rel=1e-4)
