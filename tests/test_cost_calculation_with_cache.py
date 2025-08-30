"""Test cost calculations with cached token discounts."""

from typing import cast

import pytest

# Type definitions for telemetry structures
AttemptDict = dict[str, int | str | float]
NodeDict = dict[str, str | list[AttemptDict]]
DocumentDict = dict[str, list[NodeDict]]
TelemetryDict = dict[str, DocumentDict]
NodeCostDict = dict[str, float | int]
CostAnalysisDict = dict[str, dict[str, NodeCostDict] | dict[str, float | int]]

# Test-specific pricing data for backwards compatibility
MODEL_PRICING = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006, "cache_discount": 0.5},
    "gpt-4o": {"input": 0.0025, "output": 0.01, "cache_discount": 0.5},
    "gpt-4": {"input": 0.03, "output": 0.06, "cache_discount": 0.5},
}


def calculate_summary_attempt_cost(
    attempt: dict[str, object], pricing: dict[str, dict[str, float]]
) -> float:
    """Calculate the cost of a single summary attempt with cache discount support."""
    model = attempt.get("model", "")

    # Handle passthrough model (no cost)
    if model == "passthrough":
        return 0.0

    # Get pricing for the model
    if model not in pricing:
        return 0.0
    model_pricing = pricing[model]

    # Get token counts
    prompt_tokens = attempt.get("prompt_tokens", 0)
    cached_tokens = attempt.get("cached_tokens", 0)
    completion_tokens = attempt.get("completion_tokens", 0)

    # Calculate non-cached prompt tokens
    non_cached_tokens = prompt_tokens - cached_tokens

    # Get cache discount (default to 1.0 = no discount)
    cache_discount = model_pricing.get("cache_discount", 1.0)

    # Calculate cost
    cost = (
        (cached_tokens * model_pricing["input"] * cache_discount)
        + (non_cached_tokens * model_pricing["input"])
        + (completion_tokens * model_pricing["output"])
    ) / 1000

    return float(cost)


def analyze_summary_costs(telemetry_data: TelemetryDict) -> CostAnalysisDict:
    """Analyze summary costs from telemetry data."""
    pricing = MODEL_PRICING

    by_node: dict[str, NodeCostDict] = {}
    total_dict: dict[str, float | int] = {"total_cost": 0.0, "total_attempts": 0}
    cache_dict: dict[str, float | int] = {
        "total_cached_tokens": 0,
        "total_prompt_tokens": 0,
        "cache_rate": 0.0,
    }

    for doc_id, doc_data in telemetry_data.get("documents", {}).items():
        for node in doc_data.get("nodes", []):
            node_id = cast(str, node["node_id"])
            attempts = cast(list[AttemptDict], node.get("summary_attempts", []))

            node_cost = 0.0
            node_cost_without_cache = 0.0

            for attempt in attempts:
                # Calculate actual cost
                cost = calculate_summary_attempt_cost(attempt, pricing)
                node_cost += cost

                # Calculate cost without cache for comparison
                attempt_no_cache = attempt.copy()
                attempt_no_cache["cached_tokens"] = 0
                cost_without_cache = calculate_summary_attempt_cost(
                    attempt_no_cache, pricing
                )
                node_cost_without_cache += cost_without_cache

                # Track cache efficiency
                cache_dict["total_cached_tokens"] = cast(
                    int, cache_dict["total_cached_tokens"]
                ) + cast(int, attempt.get("cached_tokens", 0))
                cache_dict["total_prompt_tokens"] = cast(
                    int, cache_dict["total_prompt_tokens"]
                ) + cast(int, attempt.get("prompt_tokens", 0))

            savings_pct = (
                ((node_cost_without_cache - node_cost) / node_cost_without_cache * 100)
                if node_cost_without_cache > 0
                else 0.0
            )
            by_node[node_id] = {
                "total_cost": node_cost,
                "cost_without_cache": node_cost_without_cache,
                "cache_savings": node_cost_without_cache - node_cost,
                "cache_savings_pct": savings_pct,
                "attempts": len(attempts),
            }

            total_dict["total_cost"] = cast(float, total_dict["total_cost"]) + node_cost
            total_dict["total_attempts"] = cast(
                int, total_dict["total_attempts"]
            ) + len(attempts)

    # Calculate overall cache rate
    if cast(int, cache_dict["total_prompt_tokens"]) > 0:
        cache_dict["cache_rate"] = cast(int, cache_dict["total_cached_tokens"]) / cast(
            int, cache_dict["total_prompt_tokens"]
        )

    costs: CostAnalysisDict = {
        "by_node": by_node,
        "total": total_dict,
        "cache_efficiency": cache_dict,
    }
    return costs


def test_calculate_cost_with_cached_tokens() -> None:
    """Test that cached tokens receive appropriate discount."""
    # Use pricing constants
    pricing = MODEL_PRICING

    # Test attempt with cached tokens
    attempt = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 1500,
        "cached_tokens": 1200,  # 80% cached
        "completion_tokens": 100,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)

    # Expected calculation:
    # - 1200 cached tokens at 50% discount
    # - 300 non-cached tokens at full price
    # - 100 completion tokens at full price
    model_pricing = pricing["gpt-4o-mini"]
    cache_discount = model_pricing.get("cache_discount", 1.0)

    expected_cost = (
        (1200 * model_pricing["input"] * cache_discount)  # Cached input
        + (300 * model_pricing["input"])  # Non-cached input
        + (100 * model_pricing["output"])  # Completion
    ) / 1000

    assert cost == pytest.approx(expected_cost, rel=1e-6)


def test_calculate_cost_without_cached_tokens() -> None:
    """Test backward compatibility when cached_tokens is not present."""
    pricing = MODEL_PRICING

    # Old-style attempt without cached_tokens field
    attempt = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 1500,
        "completion_tokens": 100,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)

    # Should treat all tokens as non-cached
    model_pricing = pricing["gpt-4o-mini"]
    expected_cost = (
        (1500 * model_pricing["input"]) + (100 * model_pricing["output"])
    ) / 1000

    assert cost == pytest.approx(expected_cost, rel=1e-6)


def test_calculate_cost_with_zero_cached_tokens() -> None:
    """Test that zero cached tokens works correctly."""
    pricing = MODEL_PRICING

    attempt = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 1500,
        "cached_tokens": 0,  # Explicitly zero
        "completion_tokens": 100,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)

    # All tokens at full price
    model_pricing = pricing["gpt-4o-mini"]
    expected_cost = (
        (1500 * model_pricing["input"]) + (100 * model_pricing["output"])
    ) / 1000

    assert cost == pytest.approx(expected_cost, rel=1e-6)


def test_calculate_cost_with_high_cache_rate() -> None:
    """Test cost savings with very high cache hit rate."""
    pricing = MODEL_PRICING

    # 95% cache hit rate
    attempt = {
        "model": "gpt-4o",  # More expensive model
        "prompt_tokens": 2000,
        "cached_tokens": 1900,
        "completion_tokens": 150,
    }

    cost_with_cache = calculate_summary_attempt_cost(attempt, pricing)

    # Compare to cost without caching
    attempt_no_cache = {
        "model": "gpt-4o",
        "prompt_tokens": 2000,
        "cached_tokens": 0,
        "completion_tokens": 150,
    }

    cost_without_cache = calculate_summary_attempt_cost(attempt_no_cache, pricing)

    # Should see significant savings
    savings_ratio = 1 - (cost_with_cache / cost_without_cache)

    # With 95% cache rate and 50% discount, expect ~47.5% savings on input tokens
    # (0.95 * 0.5 = 0.475 discount on input portion)
    assert savings_ratio > 0.3, f"Expected >30% savings, got {savings_ratio:.1%}"


def test_analyze_summary_costs_with_cached_tokens() -> None:
    """Test that analyze_summary_costs correctly aggregates cached token costs."""
    telemetry_data = {
        "documents": {
            "doc1": {
                "nodes": [
                    {
                        "node_id": "node1",
                        "summary_attempts": [
                            {
                                "model": "gpt-4o-mini",
                                "prompt_tokens": 1000,
                                "cached_tokens": 0,
                                "completion_tokens": 100,
                                "status": "rejected_over",
                            },
                            {
                                "model": "gpt-4o-mini",
                                "prompt_tokens": 1200,
                                "cached_tokens": 900,  # 75% cached on retry
                                "completion_tokens": 95,
                                "status": "accepted",
                            },
                        ],
                    },
                    {
                        "node_id": "node2",
                        "summary_attempts": [
                            {
                                "model": "gpt-4o-mini",
                                "prompt_tokens": 1000,
                                "cached_tokens": 0,
                                "completion_tokens": 100,
                                "status": "accepted",
                            }
                        ],
                    },
                ],
            }
        }
    }

    costs = analyze_summary_costs(telemetry_data)

    # Should have cost data for both nodes
    by_node = cast(dict[str, NodeCostDict], costs["by_node"])
    assert "node1" in by_node
    assert "node2" in by_node

    # Node1 should show savings from caching
    node1_cost = by_node["node1"]
    assert cast(int, node1_cost["attempts"]) == 2
    assert cast(float, node1_cost["total_cost"]) > 0

    # Verify cache efficiency metrics
    assert "cache_efficiency" in costs
    cache_efficiency = cast(dict[str, float | int], costs["cache_efficiency"])
    assert cast(int, cache_efficiency["total_cached_tokens"]) == 900
    assert cast(int, cache_efficiency["total_prompt_tokens"]) >= 3200

    # Cache rate should be meaningful
    cache_rate = cast(float, cache_efficiency["cache_rate"])
    assert 0.2 < cache_rate < 0.3  # ~900/3200 ≈ 28%


def test_cost_calculation_with_missing_model() -> None:
    """Test graceful handling when model pricing is not available."""
    pricing = {
        "gpt-4o-mini": {
            "input": 0.00015,
            "output": 0.0006,
            "cache_discount": 0.5,
        }
    }

    # Attempt with unknown model
    attempt = {
        "model": "claude-3-opus",  # Not in pricing
        "prompt_tokens": 1000,
        "cached_tokens": 500,
        "completion_tokens": 100,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)

    # Should return 0 or handle gracefully
    assert cost == 0


def test_cost_calculation_with_passthrough_model() -> None:
    """Test that passthrough summaries have zero cost."""
    pricing = MODEL_PRICING

    attempt = {
        "model": "passthrough",
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)
    assert cost == 0


def test_cost_savings_calculation() -> None:
    """Test calculation of cost savings from caching."""
    telemetry_data = {
        "documents": {
            "doc1": {
                "nodes": [
                    {
                        "node_id": "node1",
                        "summary_attempts": [
                            {
                                "model": "gpt-4o",
                                "prompt_tokens": 2000,
                                "cached_tokens": 1800,  # 90% cached
                                "completion_tokens": 200,
                                "status": "accepted",
                            }
                        ],
                    }
                ],
            }
        }
    }

    costs = analyze_summary_costs(telemetry_data)

    # Calculate expected savings
    # With 90% cache rate at 50% discount, save 45% on prompt tokens
    by_node = cast(dict[str, NodeCostDict], costs["by_node"])
    node_cost = by_node["node1"]
    assert "cost_without_cache" in node_cost
    assert "cache_savings" in node_cost

    savings_pct = cast(float, node_cost["cache_savings_pct"])
    assert savings_pct > 30  # Should save at least 30%


def test_aggregate_costs_across_documents() -> None:
    """Test that costs are correctly aggregated across multiple documents."""
    telemetry_data = {
        "documents": {
            "doc1": {
                "nodes": [
                    {
                        "node_id": "d1_n1",
                        "summary_attempts": [
                            {
                                "model": "gpt-4o-mini",
                                "prompt_tokens": 1000,
                                "cached_tokens": 500,
                                "completion_tokens": 100,
                                "status": "accepted",
                            }
                        ],
                    }
                ],
            },
            "doc2": {
                "nodes": [
                    {
                        "node_id": "d2_n1",
                        "summary_attempts": [
                            {
                                "model": "gpt-4o-mini",
                                "prompt_tokens": 1200,
                                "cached_tokens": 1000,
                                "completion_tokens": 120,
                                "status": "accepted",
                            }
                        ],
                    }
                ],
            },
        }
    }

    costs = analyze_summary_costs(telemetry_data)

    # Check total costs
    total_dict = cast(dict[str, float | int], costs["total"])
    assert cast(float, total_dict["total_cost"]) > 0
    assert cast(int, total_dict["total_attempts"]) == 2

    # Check cache efficiency across all documents
    cache_efficiency = cast(dict[str, float | int], costs["cache_efficiency"])
    assert cast(int, cache_efficiency["total_cached_tokens"]) == 1500
    assert cast(int, cache_efficiency["total_prompt_tokens"]) == 2200
