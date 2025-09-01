"""Telemetry collector for retrieval operations."""

import time

from ragzoom.telemetry_query import QueryTelemetry


class TelemetryCollector:
    """Collects telemetry data during retrieval operations."""

    def __init__(self) -> None:
        """Initialize telemetry collector."""
        self.telemetry: QueryTelemetry | None = None
        self.phase_start: float = 0.0

    def start_query(
        self,
        query_text: str,
        num_seeds: int | None,
        budget_tokens: int | None,
        document_id: str | None,
    ) -> QueryTelemetry:
        """Start telemetry collection for a query.

        Args:
            query_text: Query text
            num_seeds: Number of seeds requested
            budget_tokens: Token budget
            document_id: Document ID

        Returns:
            New QueryTelemetry instance
        """
        self.telemetry = QueryTelemetry(
            query_text=query_text,
            num_seeds=num_seeds,
            budget_tokens=budget_tokens,
            document_id=document_id,
        )
        return self.telemetry

    def start_phase(self) -> None:
        """Start timing a new phase."""
        self.phase_start = time.perf_counter()

    def end_phase(self, phase_name: str) -> None:
        """End timing for a phase and record it.

        Args:
            phase_name: Name of the phase to record
        """
        if self.telemetry:
            elapsed = time.perf_counter() - self.phase_start
            setattr(self.telemetry, f"{phase_name}_time", elapsed)

    def record_metric(self, metric_name: str, value: int | float | str) -> None:
        """Record a metric value.

        Args:
            metric_name: Name of the metric
            value: Value to record
        """
        if self.telemetry:
            setattr(self.telemetry, metric_name, value)

    def finalize(self) -> QueryTelemetry | None:
        """Finalize telemetry collection.

        Returns:
            Completed telemetry or None
        """
        if self.telemetry:
            self.telemetry.end_time = time.perf_counter()
        return self.telemetry
