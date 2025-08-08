"""Test cost calculations with cached token discounts."""

import json
from pathlib import Path
from typing import Any

import pytest


def calculate_summary_attempt_cost(attempt: dict[str, Any], pricing: dict) -> float:
    """Calculate the cost of a single summary attempt with cache discount support."""
    model = attempt.get("model", "")

    # Handle passthrough model (no cost)
    if model == "passthrough":
        return 0.0

    # Get pricing for the model
    model_pricing = pricing.get("llms", {}).get(model)
    if not model_pricing:
        return 0.0

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

    return cost


def analyze_summary_costs(telemetry_data: dict) -> dict:
    """Analyze summary costs from telemetry data."""
    pricing_path = Path(__file__).parent.parent / "ragzoom" / "pricing.json"
    with open(pricing_path) as f:
        pricing = json.load(f)

    costs = {
        "by_node": {},
        "total": {"total_cost": 0, "total_attempts": 0},
        "cache_efficiency": {
            "total_cached_tokens": 0,
            "total_prompt_tokens": 0,
            "cache_rate": 0,
        },
    }

    for doc_id, doc_data in telemetry_data.get("documents", {}).items():
        for node in doc_data.get("nodes", []):
            node_id = node["node_id"]
            attempts = node.get("summary_attempts", [])

            node_cost = 0
            node_cost_without_cache = 0

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
                costs["cache_efficiency"]["total_cached_tokens"] += attempt.get(
                    "cached_tokens", 0
                )
                costs["cache_efficiency"]["total_prompt_tokens"] += attempt.get(
                    "prompt_tokens", 0
                )

            costs["by_node"][node_id] = {
                "total_cost": node_cost,
                "cost_without_cache": node_cost_without_cache,
                "cache_savings": node_cost_without_cache - node_cost,
                "cache_savings_pct": (
                    (
                        (node_cost_without_cache - node_cost)
                        / node_cost_without_cache
                        * 100
                    )
                    if node_cost_without_cache > 0
                    else 0
                ),
                "attempts": len(attempts),
            }

            costs["total"]["total_cost"] += node_cost
            costs["total"]["total_attempts"] += len(attempts)

    # Calculate overall cache rate
    if costs["cache_efficiency"]["total_prompt_tokens"] > 0:
        costs["cache_efficiency"]["cache_rate"] = (
            costs["cache_efficiency"]["total_cached_tokens"]
            / costs["cache_efficiency"]["total_prompt_tokens"]
        )

    return costs


def test_calculate_cost_with_cached_tokens():
    """Test that cached tokens receive appropriate discount."""
    # Load pricing (will be updated to include cache_discount)
    pricing_path = Path(__file__).parent.parent / "ragzoom" / "pricing.json"
    with open(pricing_path) as f:
        pricing = json.load(f)

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
    model_pricing = pricing["llms"]["gpt-4o-mini"]
    cache_discount = model_pricing.get("cache_discount", 1.0)

    expected_cost = (
        (1200 * model_pricing["input"] * cache_discount)  # Cached input
        + (300 * model_pricing["input"])  # Non-cached input
        + (100 * model_pricing["output"])  # Completion
    ) / 1000

    assert cost == pytest.approx(expected_cost, rel=1e-6)


def test_calculate_cost_without_cached_tokens():
    """Test backward compatibility when cached_tokens is not present."""
    pricing_path = Path(__file__).parent.parent / "ragzoom" / "pricing.json"
    with open(pricing_path) as f:
        pricing = json.load(f)

    # Old-style attempt without cached_tokens field
    attempt = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 1500,
        "completion_tokens": 100,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)

    # Should treat all tokens as non-cached
    model_pricing = pricing["llms"]["gpt-4o-mini"]
    expected_cost = (
        (1500 * model_pricing["input"]) + (100 * model_pricing["output"])
    ) / 1000

    assert cost == pytest.approx(expected_cost, rel=1e-6)


def test_calculate_cost_with_zero_cached_tokens():
    """Test that zero cached tokens works correctly."""
    pricing_path = Path(__file__).parent.parent / "ragzoom" / "pricing.json"
    with open(pricing_path) as f:
        pricing = json.load(f)

    attempt = {
        "model": "gpt-4o-mini",
        "prompt_tokens": 1500,
        "cached_tokens": 0,  # Explicitly zero
        "completion_tokens": 100,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)

    # All tokens at full price
    model_pricing = pricing["llms"]["gpt-4o-mini"]
    expected_cost = (
        (1500 * model_pricing["input"]) + (100 * model_pricing["output"])
    ) / 1000

    assert cost == pytest.approx(expected_cost, rel=1e-6)


def test_calculate_cost_with_high_cache_rate():
    """Test cost savings with very high cache hit rate."""
    pricing_path = Path(__file__).parent.parent / "ragzoom" / "pricing.json"
    with open(pricing_path) as f:
        pricing = json.load(f)

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


def test_analyze_summary_costs_with_cached_tokens():
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
    assert "node1" in costs["by_node"]
    assert "node2" in costs["by_node"]

    # Node1 should show savings from caching
    node1_cost = costs["by_node"]["node1"]
    assert node1_cost["attempts"] == 2
    assert node1_cost["total_cost"] > 0

    # Verify cache efficiency metrics
    assert "cache_efficiency" in costs
    assert costs["cache_efficiency"]["total_cached_tokens"] == 900
    assert costs["cache_efficiency"]["total_prompt_tokens"] >= 3200

    # Cache rate should be meaningful
    cache_rate = costs["cache_efficiency"]["cache_rate"]
    assert 0.2 < cache_rate < 0.3  # ~900/3200 ≈ 28%


def test_cost_calculation_with_missing_model():
    """Test graceful handling when model pricing is not available."""
    pricing = {
        "llms": {
            "gpt-4o-mini": {
                "input": 0.00015,
                "output": 0.0006,
                "cache_discount": 0.5,
            }
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


def test_cost_calculation_with_passthrough_model():
    """Test that passthrough summaries have zero cost."""
    pricing_path = Path(__file__).parent.parent / "ragzoom" / "pricing.json"
    with open(pricing_path) as f:
        pricing = json.load(f)

    attempt = {
        "model": "passthrough",
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
    }

    cost = calculate_summary_attempt_cost(attempt, pricing)
    assert cost == 0


def test_cost_savings_calculation():
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
    node_cost = costs["by_node"]["node1"]
    assert "cost_without_cache" in node_cost
    assert "cache_savings" in node_cost

    savings_pct = node_cost["cache_savings_pct"]
    assert savings_pct > 30  # Should save at least 30%


def test_aggregate_costs_across_documents():
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
    assert costs["total"]["total_cost"] > 0
    assert costs["total"]["total_attempts"] == 2

    # Check cache efficiency across all documents
    assert costs["cache_efficiency"]["total_cached_tokens"] == 1500
    assert costs["cache_efficiency"]["total_prompt_tokens"] == 2200
