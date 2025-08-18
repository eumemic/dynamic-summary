"""Telemetry data collection for RagZoom query operations.

This module provides data structures and collection mechanisms for query telemetry data.
It tracks detailed timing information for each phase of the retrieval pipeline.
"""

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class QueryTelemetry:
    """Telemetry for a complete query operation.

    Tracks timing for each phase of the retrieval pipeline:
    - Embedding generation
    - Initial similarity search
    - MMR diversity selection
    - Coverage map building
    - Node scoring
    - Dynamic programming tiling
    - Text assembly
    """

    # Query parameters
    query_text: str
    num_seeds: int | None
    budget_tokens: int | None
    document_id: str | None

    # Timing for each phase (in seconds)
    embedding_time: float = 0.0
    search_time: float = 0.0
    mmr_time: float = 0.0
    coverage_map_time: float = 0.0
    scoring_time: float = 0.0
    dp_time: float = 0.0
    assembly_time: float = 0.0

    # Overall timing
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0

    # Result metrics
    seeds_requested: int = 0
    seeds_found: int = 0
    candidates_retrieved: int = 0
    coverage_size: int = 0
    tiling_size: int = 0
    output_tokens: int = 0

    # Model information
    embedding_model: str = ""

    @property
    def total_time(self) -> float:
        """Total query execution time."""
        if self.end_time > 0:
            return self.end_time - self.start_time
        return 0.0

    @property
    def retrieval_time(self) -> float:
        """Time spent in retrieval phases (embedding + search + MMR + coverage)."""
        return (
            self.embedding_time
            + self.search_time
            + self.mmr_time
            + self.coverage_map_time
        )

    @property
    def processing_time(self) -> float:
        """Time spent in processing phases (scoring + DP + assembly)."""
        return self.scoring_time + self.dp_time + self.assembly_time

    def to_dict(self) -> dict[str, Any]:
        """Convert telemetry to dictionary for JSON serialization."""
        return {
            "query_text": self.query_text,
            "num_seeds": self.num_seeds,
            "budget_tokens": self.budget_tokens,
            "document_id": self.document_id,
            "embedding_model": self.embedding_model,
            "timings": {
                "embedding_time": self.embedding_time,
                "search_time": self.search_time,
                "mmr_time": self.mmr_time,
                "coverage_map_time": self.coverage_map_time,
                "scoring_time": self.scoring_time,
                "dp_time": self.dp_time,
                "assembly_time": self.assembly_time,
                "total_time": self.total_time,
                "retrieval_time": self.retrieval_time,
                "processing_time": self.processing_time,
            },
            "metrics": {
                "seeds_requested": self.seeds_requested,
                "seeds_found": self.seeds_found,
                "candidates_retrieved": self.candidates_retrieved,
                "coverage_size": self.coverage_size,
                "tiling_size": self.tiling_size,
                "output_tokens": self.output_tokens,
            },
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


@dataclass
class QueryPhaseMetrics:
    """Aggregated metrics for query performance analysis."""

    # Phase timing breakdown (median values across queries)
    phase_breakdown: dict[str, float]

    # Efficiency metrics
    seeds_utilization: float  # seeds_found / seeds_requested
    budget_utilization: float  # output_tokens / budget_tokens
    coverage_efficiency: float  # tiling_size / coverage_size

    # Latency percentiles
    p50_latency: float
    p95_latency: float
    p99_latency: float

    # Query count
    query_count: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "phase_breakdown": self.phase_breakdown,
            "efficiency": {
                "seeds_utilization": self.seeds_utilization,
                "budget_utilization": self.budget_utilization,
                "coverage_efficiency": self.coverage_efficiency,
            },
            "latency_percentiles": {
                "p50": self.p50_latency,
                "p95": self.p95_latency,
                "p99": self.p99_latency,
            },
            "query_count": self.query_count,
        }
