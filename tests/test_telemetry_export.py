"""Tests for document-scoped telemetry logging and export."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from ragzoom.config import IndexConfig
from ragzoom.server.run_manager import TelemetryRunManager
from ragzoom.telemetry_export import (
    TelemetryExportError,
    export_document_telemetry,
)
from ragzoom.telemetry_log import DocumentTelemetryLog


@pytest.mark.asyncio
async def test_run_manager_writes_document_telemetry(tmp_path: Path) -> None:
    """TelemetryRunManager should persist events consumable by the exporter."""

    log = DocumentTelemetryLog(tmp_path)
    manager = TelemetryRunManager(IndexConfig.load(), telemetry_log=log)

    context = await manager.start_run(
        "doc",
        collect=True,
        source_tokens=128,
        document_path=None,
        replace_existing=True,
    )
    assert context.telemetry_collector is not None

    # Simulate node creation during the run
    collector = context.telemetry_collector
    collector.track_node_created("node-1", height=1, span=(0, 64))

    context.register_append_outcome(
        span_start=0,
        span_end=64,
        mutated_nodes=1,
        new_leaves=1,
        previous_leaf_count=0,
        total_leaves=1,
    )
    context.register_summary_node("node-1")

    await manager.record_node_committed(
        context,
        node_id="node-1",
        height=1,
        span_start=0,
        span_end=64,
    )

    await manager.complete_run(context.run_id)

    telemetry = export_document_telemetry(log, "doc")
    assert telemetry["document_id"] == "doc"
    assert telemetry["append_metadata"]["mutated_nodes"] == 1
    assert telemetry["append_metadata"]["leaf_delta"] == 1

    nodes = telemetry["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "node-1"
    assert tuple(nodes[0]["span"]) == (0, 64)

    history = cast(list[dict[str, object]], telemetry.get("append_history", []))
    assert history and history[0].get("status") == "completed"


def test_export_requires_metadata(tmp_path: Path) -> None:
    """Exporter should fail gracefully when metadata is missing."""

    log = DocumentTelemetryLog(tmp_path)
    with pytest.raises(TelemetryExportError):
        export_document_telemetry(log, "missing-doc")
