"""Telemetry data collection for RagZoom query operations.

This module provides data structures and collection mechanisms for query telemetry data.
It tracks detailed timing information for each phase of the retrieval pipeline.
"""

import time
from dataclasses import dataclass, field

from typing_extensions import TypedDict


@dataclass
class QueryTelemetry:
    """Telemetry for a complete query operation.

    Tracks timing for each phase of the retrieval pipeline:
    - Embedding generation
    - Initial similarity search
    - MMR diversity selection
    - Coverage map building
    - Node scoring
    - Greedy tiling
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
    tiling_time: float = 0.0
    assembly_time: float = 0.0

    # Overall timing
    start_time: float = field(default_factory=time.perf_counter)
    end_time: float = 0.0

    # Result metrics
    seeds_requested: int = 0
    seeds_found: int = 0
    candidates_retrieved: int = 0
    candidates_filtered: int = 0
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
        return self.scoring_time + self.tiling_time + self.assembly_time

    def to_dict(self) -> "QueryTelemetryDict":
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
                "tiling_time": self.tiling_time,
                "assembly_time": self.assembly_time,
                "total_time": self.total_time,
                "retrieval_time": self.retrieval_time,
                "processing_time": self.processing_time,
            },
            "metrics": {
                "seeds_requested": self.seeds_requested,
                "seeds_found": self.seeds_found,
                "candidates_retrieved": self.candidates_retrieved,
                "candidates_filtered": self.candidates_filtered,
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

    # Phase variance data (MAD values for dynamic thresholds)
    phase_variance: dict[str, float]

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

    def to_dict(self) -> "QueryPhaseMetricsDict":
        """Convert to dictionary for reporting."""
        return {
            "phase_breakdown": self.phase_breakdown,
            "phase_variance": self.phase_variance,
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


# TypedDict definitions for return values
class QueryTimingsDict(TypedDict):
    """Type definition for query timing data."""

    embedding_time: float
    search_time: float
    mmr_time: float
    coverage_map_time: float
    scoring_time: float
    tiling_time: float
    assembly_time: float
    total_time: float
    retrieval_time: float
    processing_time: float


class QueryMetricsDict(TypedDict):
    """Type definition for query metrics data."""

    seeds_requested: int
    seeds_found: int
    candidates_retrieved: int
    candidates_filtered: int
    coverage_size: int
    tiling_size: int
    output_tokens: int


class QueryTelemetryDict(TypedDict):
    """Type definition for QueryTelemetry.to_dict() return value."""

    query_text: str
    num_seeds: int | None
    budget_tokens: int | None
    document_id: str | None
    embedding_model: str
    timings: QueryTimingsDict
    metrics: QueryMetricsDict
    start_time: float
    end_time: float


class QueryEfficiencyDict(TypedDict):
    """Type definition for query efficiency metrics."""

    seeds_utilization: float
    budget_utilization: float
    coverage_efficiency: float


class QueryLatencyPercentilesDict(TypedDict):
    """Type definition for query latency percentiles."""

    p50: float
    p95: float
    p99: float


class QueryPhaseMetricsDict(TypedDict):
    """Type definition for QueryPhaseMetrics.to_dict() return value."""

    phase_breakdown: dict[str, float]
    phase_variance: dict[str, float]
    efficiency: QueryEfficiencyDict
    latency_percentiles: QueryLatencyPercentilesDict
    query_count: int
