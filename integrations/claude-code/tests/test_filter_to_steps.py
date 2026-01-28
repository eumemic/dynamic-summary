"""Tests for step filtering in transcript sync.

Tests the _should_include_record() helper which determines if a JSONL record
should become a Step for indexing.
"""

from __future__ import annotations


class TestShouldIncludeRecord:
    """Tests for _should_include_record() helper function.

    This function determines which records should become Steps:
    - Included: user and assistant messages
    - Excluded: queue-operation, isCompactSummary=True, isMeta=True
    """

    def test_includes_user_message(self) -> None:
        """Regular user messages are included."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "user",
            "timestamp": "2024-01-21T14:30:00Z",
            "message": {"content": "Hello"},
        }
        assert _should_include_record(record) is True

    def test_includes_assistant_message(self) -> None:
        """Regular assistant messages are included."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "assistant",
            "timestamp": "2024-01-21T14:30:05Z",
            "message": {"content": [{"type": "text", "text": "Hi!"}]},
        }
        assert _should_include_record(record) is True

    def test_includes_tool_use_result(self) -> None:
        """Tool results (user messages with toolUseResult) are included as their own steps."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "user",
            "timestamp": "2024-01-21T14:30:03Z",
            "toolUseResult": {"type": "success"},
            "message": {"content": "file contents here"},
        }
        assert _should_include_record(record) is True

    def test_excludes_queue_operation(self) -> None:
        """Queue operations (type=queue-operation) are excluded."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "queue-operation",
            "timestamp": "2024-01-21T14:30:02Z",
        }
        assert _should_include_record(record) is False

    def test_excludes_compaction_summary(self) -> None:
        """Compaction summaries (isCompactSummary=True) are excluded."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "assistant",
            "timestamp": "2024-01-21T14:30:05Z",
            "isCompactSummary": True,
            "message": {"content": [{"type": "text", "text": "Summary..."}]},
        }
        assert _should_include_record(record) is False

    def test_excludes_meta_record(self) -> None:
        """Meta records (isMeta=True) are excluded."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "user",
            "timestamp": "2024-01-21T14:30:01Z",
            "isMeta": True,
            "message": {"content": "[Skill expansion: 20MB of docs...]"},
        }
        assert _should_include_record(record) is False

    def test_excludes_unknown_type(self) -> None:
        """Records with unknown type (not user/assistant) are excluded."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "system",
            "timestamp": "2024-01-21T14:30:00Z",
        }
        assert _should_include_record(record) is False

    def test_excludes_record_without_type(self) -> None:
        """Records without type field are excluded."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "timestamp": "2024-01-21T14:30:00Z",
            "message": {"content": "Hello"},
        }
        assert _should_include_record(record) is False

    def test_user_compaction_summary_excluded(self) -> None:
        """User messages that are compaction summaries are excluded."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "user",
            "timestamp": "2024-01-21T14:30:00Z",
            "isCompactSummary": True,
            "message": {"content": "Compacted user content..."},
        }
        assert _should_include_record(record) is False

    def test_assistant_meta_record_excluded(self) -> None:
        """Assistant messages that are meta records are excluded."""
        from ragzoom_claude_code.transcript_sync import _should_include_record

        record: dict[str, object] = {
            "uuid": "msg1",
            "type": "assistant",
            "timestamp": "2024-01-21T14:30:05Z",
            "isMeta": True,
            "message": {"content": [{"type": "text", "text": "Injected content..."}]},
        }
        assert _should_include_record(record) is False
