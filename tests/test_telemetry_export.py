"""Tests for document-scoped telemetry logging and export."""

from __future__ import annotations

import time
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

    start = time.time()
    collector.record_chunk_split_start(
        start_time=start,
        new_text_chars=64,
        existing_tail_chars=0,
        combined_chars=64,
    )
    collector.record_chunk_split_end(
        end_time=start + 0.25,
        chunk_count=4,
        total_tokens=256,
    )

    await manager.log_chunk_event(
        context,
        event="chunk_split_started",
        new_text_chars=64,
        existing_tail_chars=0,
        combined_chars=64,
    )
    await manager.log_chunk_event(
        context,
        event="chunk_split_completed",
        chunk_count=4,
        duration=0.25,
        total_tokens=256,
    )

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

    telemetry = export_document_telemetry(
        log,
        "doc",
        active_nodes={"node-1": {"height": 1, "span": (0, 64)}},
    )
    assert telemetry["document_id"] == "doc"
    assert telemetry["append_metadata"]["mutated_nodes"] == 1
    assert telemetry["append_metadata"]["leaf_delta"] == 1

    nodes = telemetry["nodes"]
    assert len(nodes) == 1
    assert nodes[0]["node_id"] == "node-1"
    assert tuple(nodes[0]["span"]) == (0, 64)

    chunk_split = telemetry.get("chunk_split")
    assert chunk_split is not None
    assert chunk_split["chunk_count"] == 4
    assert chunk_split["total_tokens"] == 256

    history = cast(list[dict[str, object]], telemetry.get("append_history", []))
    assert history and "chunk_split" in history[0]

    history = cast(list[dict[str, object]], telemetry.get("append_history", []))
    assert history and history[0].get("status") == "completed"


def test_export_requires_metadata(tmp_path: Path) -> None:
    """Exporter should fail gracefully when metadata is missing."""

    log = DocumentTelemetryLog(tmp_path)
    with pytest.raises(TelemetryExportError):
        export_document_telemetry(log, "missing-doc", active_nodes={})


@pytest.mark.asyncio
async def test_export_filters_and_updates_active_nodes(tmp_path: Path) -> None:
    """Exporter should align node data with active document state."""

    log = DocumentTelemetryLog(tmp_path)
    metadata = {
        "format_version": "4.3",
        "document_id": "doc",
        "source_document_tokens": 0,
    }
    await log.ensure_metadata("doc", metadata, reset=True)

    await log.append_event(
        "doc",
        {
            "event": "append_started",
            "run_id": "run-1",
            "append_id": "run-1",
        },
    )
    await log.append_event(
        "doc",
        {
            "event": "node_committed",
            "run_id": "run-1",
            "append_id": "run-1",
            "node_id": "node-old",
            "height": 1,
            "span_start": 0,
            "span_end": 10,
            "timestamp": 1.0,
        },
    )
    await log.append_event(
        "doc",
        {
            "event": "append_completed",
            "run_id": "run-1",
            "append_id": "run-1",
            "duration": 1.0,
            "outcome": {
                "mutated_nodes": 2,
                "new_leaves": 2,
                "leaf_delta": 2,
                "nodes": [
                    {
                        "node_id": "node-old",
                        "height": 1,
                        "created_at": 0.5,
                        "span": [0, 10],
                    },
                    {
                        "node_id": "node-stale",
                        "height": 0,
                        "created_at": 0.6,
                        "span": [10, 20],
                    },
                ],
            },
        },
    )
    await log.append_event(
        "doc",
        {
            "event": "node_committed",
            "run_id": "run-2",
            "append_id": "run-2",
            "node_id": "node-old",
            "height": 2,
            "span_start": 0,
            "span_end": 15,
            "timestamp": 2.0,
        },
    )
    await log.append_event(
        "doc",
        {
            "event": "node_committed",
            "run_id": "run-2",
            "append_id": "run-2",
            "node_id": "node-only",
            "height": 0,
            "span_start": 20,
            "span_end": 24,
            "timestamp": 2.5,
        },
    )

    telemetry = export_document_telemetry(
        log,
        "doc",
        active_nodes={
            "node-old": {"height": 3, "span": (0, 15)},
            "node-only": {"height": 0, "span": (20, 25)},
        },
    )

    nodes = {node["node_id"]: node for node in telemetry["nodes"]}
    assert set(nodes) == {"node-old", "node-only"}
    assert tuple(nodes["node-old"]["span"]) == (0, 15)
    assert nodes["node-old"]["height"] == 3
    assert tuple(nodes["node-only"]["span"]) == (20, 25)
