"""Test that claude-transcriber is available as a dependency."""

from claude_transcriber import Transcriber


def test_claude_transcriber_import_succeeds() -> None:
    """Verify claude-transcriber package is importable.

    Confirms the claude-transcriber dependency from pyproject.toml
    is correctly installed.
    """
    assert Transcriber


def test_transcriber_has_transcribe_method() -> None:
    """Verify Transcriber.transcribe() method exists.

    The transcript sync feature depends on this method to convert
    JSONL records to human-readable text.
    """
    transcriber = Transcriber()
    assert callable(getattr(transcriber, "transcribe", None))
