"""Utilities for rendering worker progress across client and server UIs."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TextIO

from ragzoom.progress import get_progress_config


@dataclass(frozen=True)
class DocumentProgressTotals:
    """Aggregated progress numbers for a single document."""

    inflight: int
    completed: int
    total: int

    @classmethod
    def from_status_dicts(
        cls,
        doc_id: str,
        inflight_by_document: Mapping[str, int],
        completed_by_document: Mapping[str, int],
        expected_total_by_document: Mapping[str, int],
    ) -> DocumentProgressTotals:
        """Create progress totals from status dictionaries for a given document.

        Args:
            doc_id: Document identifier
            inflight_by_document: In-flight jobs per document
            completed_by_document: Completed jobs per document
            expected_total_by_document: Pre-calculated expected total per document.
        """
        inflight = inflight_by_document.get(doc_id, 0)
        completed = completed_by_document.get(doc_id, 0)
        total = expected_total_by_document.get(doc_id, 0)
        # Ensure total is at least as large as completed (handles edge cases)
        if total < completed:
            total = completed

        return cls(
            inflight=inflight,
            completed=completed,
            total=total,
        )


@lru_cache(maxsize=1)
def _resolve_tqdm() -> type | None:
    try:  # pragma: no cover - optional dependency
        from tqdm import tqdm as tqdm_class
    except ImportError:  # pragma: no cover - tqdm not installed
        return None
    return tqdm_class


class WorkerProgressDisplay:
    """Render progress for worker activity using tqdm when available.

    Parameters:
        focus_documents: When provided, restricts output to these document IDs.
        stream: Text stream for fallback text output (defaults to stderr).
        line_printer: Optional callable used for textual output; defaults to writing to
            ``stream``.
        enable_bars: Force enable/disable tqdm bars. When ``None`` (default), enable
            bars only when tqdm is available and the chosen stream supports TTY output.
    """

    def __init__(
        self,
        *,
        focus_documents: set[str] | None = None,
        stream: TextIO | None = None,
        line_printer: Callable[[str], object] | None = None,
        enable_bars: bool | None = None,
    ) -> None:
        self._focus_documents = focus_documents
        self._stream = stream if stream is not None else sys.stderr
        if line_printer is not None:
            self._print = line_printer
        else:

            def _printer(message: str) -> None:
                print(message, file=self._stream)

            self._print = _printer

        tqdm_cls = _resolve_tqdm()
        stream_supports_tty = bool(getattr(self._stream, "isatty", lambda: False)())
        global_cfg = get_progress_config()
        if enable_bars is not None:
            self._use_bars = enable_bars and tqdm_cls is not None
        else:
            self._use_bars = (
                tqdm_cls is not None
                and stream_supports_tty
                and not global_cfg.disable_bars
            )

        self._tqdm_cls = tqdm_cls
        self._bars: MutableMapping[str, object] = {}
        self._positions: MutableMapping[str, int] = {}
        self._next_position = 0
        self._last_documents: dict[str, DocumentProgressTotals] = {}
        self._last_queue_depth = 0
        self._last_inflight = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update(
        self,
        *,
        queue_depth: int,
        inflight: int,
        documents: Mapping[str, DocumentProgressTotals],
        message: str | None = None,
    ) -> None:
        """Update the progress display with the latest snapshot."""

        self._last_queue_depth = queue_depth
        self._last_inflight = inflight
        self._last_documents = dict(documents)

        focus_docs = self._focus_documents
        visible_docs: Sequence[str]
        if focus_docs is None:
            visible_docs = sorted(documents)
        else:
            visible_docs = [
                doc_id for doc_id in sorted(focus_docs) if doc_id in documents
            ]

        if self._use_bars and self._tqdm_cls is not None:
            self._sync_bars(visible_docs, documents)
        else:
            self._render_text(visible_docs, documents, message)

    def finish(self) -> None:
        """Close any open progress bars and restore the terminal state."""

        if not self._use_bars:
            return
        for bar in self._bars.values():
            close = getattr(bar, "close", None)
            if callable(close):
                close()
        self._bars.clear()

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def is_focus_complete(self) -> bool:
        """Return True when all focus documents have finished work."""

        if not self._focus_documents:
            return False
        for doc_id in self._focus_documents:
            progress = self._last_documents.get(doc_id)
            if progress is None:
                return False
            if progress.inflight > 0:
                return False
        return True

    def last_progress(self, doc_id: str) -> DocumentProgressTotals | None:
        return self._last_documents.get(doc_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _sync_bars(
        self,
        visible_docs: Sequence[str],
        documents: Mapping[str, DocumentProgressTotals],
    ) -> None:
        if self._tqdm_cls is None:
            return

        active_docs = set()
        completed_docs: list[str] = []

        for doc_id in visible_docs:
            progress = documents[doc_id]
            total = max(
                progress.total,
                progress.completed + progress.inflight,
            )
            if total <= 0:
                total = 1

            bar = self._bars.get(doc_id)
            if bar is None:
                position = self._positions.get(doc_id)
                if position is None:
                    position = self._next_position
                    self._positions[doc_id] = position
                    self._next_position += 1
                bar = self._tqdm_cls(
                    total=total,
                    desc=doc_id,
                    position=position,
                    leave=False,
                    dynamic_ncols=True,
                    miniters=1,
                )
                self._bars[doc_id] = bar
            else:
                if getattr(bar, "total", None) != total:
                    setattr(bar, "total", total)

            completed = progress.completed
            current_total = getattr(bar, "total", total)
            if completed > current_total:
                setattr(bar, "total", completed)

            current_n = getattr(bar, "n", 0)
            if completed < current_n:
                setattr(bar, "n", completed)
                refresh = getattr(bar, "refresh", None)
                if callable(refresh):
                    refresh()
            elif completed > current_n:
                update = getattr(bar, "update", None)
                if callable(update):
                    update(completed - current_n)

            postfix = f"inflight={progress.inflight}"
            set_postfix = getattr(bar, "set_postfix_str", None)
            if callable(set_postfix):
                set_postfix(postfix)

            active_docs.add(doc_id)

            if progress.inflight == 0 and completed >= getattr(bar, "total", completed):
                completed_docs.append(doc_id)

        for doc_id in completed_docs:
            bar = self._bars.pop(doc_id, None)
            if bar is not None:
                close = getattr(bar, "close", None)
                if callable(close):
                    close()

        for doc_id in list(self._bars.keys()):
            if doc_id not in active_docs:
                bar = self._bars.pop(doc_id)
                close = getattr(bar, "close", None)
                if callable(close):
                    close()

    def _render_text(
        self,
        visible_docs: Sequence[str],
        documents: Mapping[str, DocumentProgressTotals],
        message: str | None,
    ) -> None:
        focus_docs = self._focus_documents
        if message and focus_docs is None:
            self._print(message)

        if focus_docs:
            for doc_id in focus_docs:
                progress = documents.get(doc_id)
                if progress is None:
                    self._print(
                        f"{doc_id}: awaiting work (queue={self._last_queue_depth})"
                    )
                else:
                    self._print(
                        _format_progress_line(
                            doc_id,
                            progress,
                            inflight_total=self._last_inflight,
                        )
                    )
        else:
            if not visible_docs:
                self._print("No active documents")
                return
            for doc_id in visible_docs:
                progress = documents[doc_id]
                self._print(
                    _format_progress_line(
                        doc_id,
                        progress,
                        inflight_total=self._last_inflight,
                    )
                )


def _format_progress_line(
    doc_id: str,
    progress: DocumentProgressTotals,
    *,
    inflight_total: int,
) -> str:
    total = (
        progress.total if progress.total > 0 else progress.completed + progress.inflight
    )
    if total <= 0:
        total_str = "?"
    else:
        total_str = str(total)
    return (
        f"{doc_id}: completed={progress.completed}/{total_str} "
        f"inflight={progress.inflight} (workers={inflight_total})"
    )
