"""Utilities for synthesizing document-level telemetry from event logs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import cast

from ragzoom.telemetry_collection import TELEMETRY_FORMAT_VERSION
from ragzoom.telemetry_log import DocumentTelemetryLog
from ragzoom.telemetry_types import TelemetryDataDict

JsonDict = dict[str, object]


class TelemetryExportError(RuntimeError):
    """Raised when telemetry export cannot be completed."""


def synthesize_document_telemetry(
    metadata: JsonDict, events: Iterable[Mapping[str, object]]
) -> TelemetryDataDict:
    """Combine metadata and event stream into legacy telemetry payload."""

    base: JsonDict = dict(metadata)
    base.setdefault("format_version", TELEMETRY_FORMAT_VERSION)

    nodes_by_id: dict[str, JsonDict] = {}
    append_order: list[str] = []
    append_history: dict[str, JsonDict] = {}

    indexed_at = _as_float(base.get("indexed_at"))
    source_tokens = _as_int(base.get("source_document_tokens")) or 0
    last_outcome: JsonDict | None = None

    for event in events:
        event_type = _as_str(event.get("event"))
        append_id = _as_str(event.get("append_id")) or _as_str(event.get("run_id"))
        history_key = append_id or _as_str(event.get("run_id"))
        timestamp = _as_float(event.get("timestamp"))

        if event_type == "append_started":
            if history_key is None:
                continue
            if indexed_at is None and timestamp is not None:
                indexed_at = timestamp
            source_tokens = max(
                source_tokens,
                _as_int(event.get("source_tokens")) or 0,
            )
            entry = append_history.setdefault(
                history_key,
                {
                    "append_id": append_id,
                    "run_id": event.get("run_id"),
                },
            )
            entry.update(
                {
                    "status": "in_progress",
                    "started_at": timestamp,
                    "replace_existing": bool(event.get("replace_existing")),
                    "source_tokens": event.get("source_tokens"),
                }
            )
            if history_key and history_key not in append_order:
                append_order.append(history_key)
        elif event_type == "append_completed":
            if history_key is None:
                continue
            outcome = _as_mapping(event.get("outcome"))
            nodes_value = outcome.get("nodes") if outcome else None
            if isinstance(nodes_value, Sequence):
                for node_payload in nodes_value:
                    node = _as_mapping(node_payload)
                    if not node:
                        continue
                    node_id = _as_str(node.get("node_id"))
                    if node_id:
                        nodes_by_id[node_id] = dict(node)
            entry = append_history.setdefault(
                history_key,
                {
                    "append_id": append_id,
                    "run_id": event.get("run_id"),
                },
            )
            entry.update(
                {
                    "status": "completed",
                    "completed_at": timestamp,
                    "duration": event.get("duration"),
                    "mutated_nodes": outcome.get("mutated_nodes") if outcome else None,
                    "new_leaves": outcome.get("new_leaves") if outcome else None,
                    "leaf_delta": outcome.get("leaf_delta") if outcome else None,
                    "summary_nodes": outcome.get("summary_nodes") if outcome else None,
                }
            )
            last_outcome = dict(outcome) if outcome is not None else None
            if history_key and history_key not in append_order:
                append_order.append(history_key)
        elif event_type == "append_failed":
            if history_key is None:
                continue
            entry = append_history.setdefault(
                history_key,
                {
                    "append_id": append_id,
                    "run_id": event.get("run_id"),
                },
            )
            entry.update(
                {
                    "status": "failed",
                    "completed_at": timestamp,
                    "duration": event.get("duration"),
                    "error": event.get("error"),
                }
            )
            if history_key and history_key not in append_order:
                append_order.append(history_key)

    if not nodes_by_id and last_outcome is None:
        raise TelemetryExportError("No completed telemetry events were recorded")

    result: JsonDict = dict(base)
    if indexed_at is not None:
        result["indexed_at"] = indexed_at
    if source_tokens:
        result["source_document_tokens"] = source_tokens

    node_records: list[JsonDict] = list(nodes_by_id.values())
    node_records.sort(key=lambda payload: _as_float(payload.get("created_at")) or 0.0)
    result["nodes"] = node_records

    if last_outcome is not None:
        append_metadata = {
            "scope": "append",
            "span_start": last_outcome.get("span_start"),
            "span_end": last_outcome.get("span_end"),
            "mutated_nodes": last_outcome.get("mutated_nodes"),
            "summary_nodes": last_outcome.get("summary_nodes"),
            "leaf_delta": last_outcome.get("leaf_delta"),
        }
        result["append_metadata"] = {
            key: value for key, value in append_metadata.items() if value is not None
        }

    ordered_history = [
        append_history[append_id]
        for append_id in append_order
        if append_id in append_history
        and append_history[append_id].get("status") in {"completed", "failed"}
    ]
    if ordered_history:
        result["append_history"] = ordered_history

    return cast(TelemetryDataDict, result)


def export_document_telemetry(
    log: DocumentTelemetryLog, document_id: str
) -> TelemetryDataDict:
    """Load metadata and events for a document and synthesize telemetry."""

    metadata = log.read_metadata(document_id)
    if metadata is None:
        raise TelemetryExportError(
            f"No telemetry metadata recorded for document '{document_id}'"
        )

    events = list(log.replay_events(document_id))
    if not events:
        raise TelemetryExportError(
            f"Telemetry event log is empty for document '{document_id}'"
        )

    return synthesize_document_telemetry(metadata, events)


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        if isinstance(value, int | float):
            number = float(value)
        elif isinstance(value, str):
            number = float(value)
        else:
            return None
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        if isinstance(value, int | float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        return None
    except (TypeError, ValueError):
        return None


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    return None


__all__ = [
    "TelemetryExportError",
    "export_document_telemetry",
    "synthesize_document_telemetry",
]
