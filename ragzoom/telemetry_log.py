"""Document-scoped telemetry logging utilities."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from collections.abc import Iterator
from pathlib import Path
from typing import cast

JsonDict = dict[str, object]


def _sanitize_document_id(document_id: str) -> str:
    """Map a document identifier to a filesystem-safe directory name."""

    sanitized = re.sub(r"[^0-9A-Za-z._-]", "_", document_id)
    return sanitized or "document"


class DocumentTelemetryLog:
    """Persistent telemetry journal stored per document."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def ensure_metadata(
        self,
        document_id: str,
        metadata: JsonDict,
        *,
        reset: bool,
    ) -> None:
        """Ensure metadata exists and optionally reset the event log."""

        async with await self._acquire_lock(document_id):
            doc_dir = self._doc_dir(document_id)
            doc_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = doc_dir / "telemetry.meta.json"
            events_path = doc_dir / "telemetry.events.jsonl"

            if reset and events_path.exists():
                events_path.unlink()

            # Always rewrite metadata to capture latest configuration.
            await asyncio.to_thread(
                self._write_json_atomic,
                metadata_path,
                metadata,
            )

    async def append_event(self, document_id: str, event: JsonDict) -> None:
        """Append a telemetry event with an automatic timestamp."""

        payload = dict(event)
        payload.setdefault("timestamp", time.time())

        async with await self._acquire_lock(document_id):
            doc_dir = self._doc_dir(document_id)
            doc_dir.mkdir(parents=True, exist_ok=True)
            events_path = doc_dir / "telemetry.events.jsonl"

            line = json.dumps(payload, separators=(",", ":"))
            await asyncio.to_thread(self._append_line, events_path, line)

    async def clear(self, document_id: str) -> None:
        """Remove telemetry artifacts for the given document."""

        async with await self._acquire_lock(document_id):
            doc_dir = self._doc_dir(document_id)
            await asyncio.to_thread(shutil.rmtree, doc_dir, True)

    def read_metadata(self, document_id: str) -> JsonDict | None:
        """Return stored metadata for the document, if any."""

        metadata_path = self._doc_dir(document_id) / "telemetry.meta.json"
        if not metadata_path.exists():
            return None
        with metadata_path.open("r", encoding="utf-8") as fh:
            return cast(JsonDict, json.load(fh))

    def replay_events(self, document_id: str) -> Iterator[JsonDict]:
        """Yield parsed telemetry events for a document."""

        events_path = self._doc_dir(document_id) / "telemetry.events.jsonl"
        if not events_path.exists():
            return iter(())

        def _reader() -> Iterator[JsonDict]:
            with events_path.open("r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(payload, dict):
                        yield cast(JsonDict, payload)

        return _reader()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    async def _acquire_lock(self, document_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(document_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[document_id] = lock
            return lock

    def _doc_dir(self, document_id: str) -> Path:
        return self._base_dir / _sanitize_document_id(document_id)

    @staticmethod
    def _write_json_atomic(path: Path, payload: JsonDict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_suffix(path.suffix + ".tmp")
        with temp_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        temp_path.replace(path)

    @staticmethod
    def _append_line(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
            fh.write(os.linesep)


__all__ = ["DocumentTelemetryLog"]
