"""Protocol conformance tests for TranscriptSyncClient."""

from __future__ import annotations

from ragzoom_claude_code.transcript_sync import TranscriptSyncClient


def test_ragzoom_wrapper_satisfies_protocol() -> None:
    """RagZoom wrapper must satisfy the TranscriptSyncClient protocol."""
    from ragzoom.wrapper import RagZoom

    assert issubclass(RagZoom, TranscriptSyncClient)


def test_execute_sync_client_param_is_typed() -> None:
    """execute_sync must type its client parameter as TranscriptSyncClient."""
    from typing import get_type_hints

    from ragzoom_claude_code.transcript_sync import execute_sync

    hints = get_type_hints(execute_sync)
    assert hints["client"] is TranscriptSyncClient
