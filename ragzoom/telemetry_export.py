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
    metadata: JsonDict,
    events: Iterable[Mapping[str, object]],
    *,
    active_nodes: Mapping[str, Mapping[str, object]] | None = None,
) -> TelemetryDataDict:
    """Combine metadata and event stream into legacy telemetry payload."""

    base: JsonDict = dict(metadata)
    base.setdefault("format_version", TELEMETRY_FORMAT_VERSION)

    nodes_by_id: dict[str, JsonDict] = {}
    append_order: list[str] = []
    append_history: dict[str, JsonDict] = {}
    committed_nodes: dict[str, JsonDict] = {}

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
        elif event_type == "node_committed":
            node_id = _as_str(event.get("node_id"))
            if node_id is None:
                continue
            entry = committed_nodes.setdefault(node_id, {"node_id": node_id})
            height = _coerce_int(event.get("height"))
            if height is not None:
                entry["height"] = height
            span_start = _coerce_int(event.get("span_start"))
            span_end = _coerce_int(event.get("span_end"))
            if span_start is not None and span_end is not None:
                entry["span"] = (span_start, span_end)
            if "created_at" not in entry:
                timestamp = _as_float(event.get("timestamp"))
                if timestamp is not None:
                    entry["created_at"] = timestamp
            continue

    if not nodes_by_id and last_outcome is None:
        raise TelemetryExportError("No completed telemetry events were recorded")

    result: JsonDict = dict(base)
    if indexed_at is not None:
        result["indexed_at"] = indexed_at
    if source_tokens:
        result["source_document_tokens"] = source_tokens

    node_records = _merge_node_payloads(nodes_by_id, committed_nodes, active_nodes)
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
    log: DocumentTelemetryLog,
    document_id: str,
    *,
    active_nodes: Mapping[str, Mapping[str, object]] | None = None,
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

    return synthesize_document_telemetry(
        metadata,
        events,
        active_nodes=active_nodes,
    )


def _merge_node_payloads(
    nodes_by_id: Mapping[str, JsonDict],
    committed_nodes: Mapping[str, JsonDict],
    active_nodes: Mapping[str, Mapping[str, object]] | None,
) -> list[JsonDict]:
    if active_nodes is None:
        merged: list[JsonDict] = []
        for node_id, payload in nodes_by_id.items():
            merged_payload = _apply_node_updates(
                node_id,
                payload,
                committed_nodes.get(node_id),
                None,
            )
            merged.append(merged_payload)
        merged.sort(key=lambda payload: _as_float(payload.get("created_at")) or 0.0)
        return merged

    final_nodes: list[JsonDict] = []
    for node_id, active_payload in active_nodes.items():
        merged_payload = _apply_node_updates(
            node_id,
            nodes_by_id.get(node_id),
            committed_nodes.get(node_id),
            active_payload,
        )
        final_nodes.append(merged_payload)

    final_nodes.sort(key=lambda payload: _as_float(payload.get("created_at")) or 0.0)
    return final_nodes


def _apply_node_updates(
    node_id: str,
    base_payload: JsonDict | None,
    committed_payload: JsonDict | None,
    active_payload: Mapping[str, object] | None,
) -> JsonDict:
    merged: JsonDict = {"node_id": node_id}
    if base_payload:
        merged.update(base_payload)

    if committed_payload:
        if "height" in committed_payload:
            merged["height"] = committed_payload["height"]
        if "span" in committed_payload:
            merged["span"] = committed_payload["span"]
        if "created_at" in committed_payload and "created_at" not in merged:
            merged["created_at"] = committed_payload["created_at"]

    if active_payload:
        height_value = active_payload.get("height")
        height = _coerce_int(height_value) if height_value is not None else None
        if height is not None:
            merged["height"] = height
        span_value = active_payload.get("span")
        active_span = _coerce_span(span_value) if span_value is not None else None
        if active_span is not None:
            merged["span"] = active_span

    existing_span = merged.get("span")
    coerced_span = _coerce_span(existing_span) if existing_span is not None else None
    if coerced_span is not None:
        merged["span"] = coerced_span
    elif existing_span is not None:
        merged.pop("span", None)

    if "height" not in merged:
        merged["height"] = 0

    if "created_at" not in merged:
        # Fallback to zero to keep sorting deterministic.
        merged["created_at"] = 0.0

    return merged


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


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    return _as_int(value)


def _coerce_span(value: object) -> tuple[int, int] | None:
    if isinstance(value, Mapping):
        start = _coerce_int(value.get("span_start"))
        end = _coerce_int(value.get("span_end"))
    elif isinstance(value, Sequence):
        sequence = list(value)
        if len(sequence) != 2:
            return None
        start = _coerce_int(sequence[0])
        end = _coerce_int(sequence[1])
    else:
        return None

    if start is None or end is None:
        return None
    return (start, end)


__all__ = [
    "TelemetryExportError",
    "export_document_telemetry",
    "synthesize_document_telemetry",
]
