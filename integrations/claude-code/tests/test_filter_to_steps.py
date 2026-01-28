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


class TestFilterToSteps:
    """Tests for filter_to_steps() function.

    This function takes (uuids, records_by_uuid) and returns list[Step]
    filtering to user/assistant messages with valid timestamps.
    """

    def test_filters_to_user_and_assistant_messages(self) -> None:
        """User and assistant messages become Steps."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            "asst1": {
                "uuid": "asst1",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi!"}]},
            },
        }
        uuids = ["user1", "asst1"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 2
        assert steps[0] == Step(uuid="user1", timestamp="2024-01-21T14:30:00Z")
        assert steps[1] == Step(uuid="asst1", timestamp="2024-01-21T14:30:05Z")

    def test_excludes_queue_operation(self) -> None:
        """Queue operations are filtered out."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            "queue1": {
                "uuid": "queue1",
                "type": "queue-operation",
                "timestamp": "2024-01-21T14:30:01Z",
            },
            "asst1": {
                "uuid": "asst1",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi!"}]},
            },
        }
        uuids = ["user1", "queue1", "asst1"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 2
        assert steps[0] == Step(uuid="user1", timestamp="2024-01-21T14:30:00Z")
        assert steps[1] == Step(uuid="asst1", timestamp="2024-01-21T14:30:05Z")

    def test_excludes_compaction_summary(self) -> None:
        """Compaction summaries (isCompactSummary=True) are filtered out."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            "compact1": {
                "uuid": "compact1",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "isCompactSummary": True,
                "message": {"content": [{"type": "text", "text": "Summary..."}]},
            },
        }
        uuids = ["user1", "compact1"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 1
        assert steps[0] == Step(uuid="user1", timestamp="2024-01-21T14:30:00Z")

    def test_excludes_meta_record(self) -> None:
        """Meta records (isMeta=True) are filtered out."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            "meta1": {
                "uuid": "meta1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:01Z",
                "isMeta": True,
                "message": {"content": "[Skill expansion...]"},
            },
        }
        uuids = ["user1", "meta1"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 1
        assert steps[0] == Step(uuid="user1", timestamp="2024-01-21T14:30:00Z")

    def test_skips_records_without_timestamp(self) -> None:
        """Records without timestamp field are skipped."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            "user2": {
                "uuid": "user2",
                "type": "user",
                # No timestamp!
                "message": {"content": "No timestamp"},
            },
        }
        uuids = ["user1", "user2"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 1
        assert steps[0] == Step(uuid="user1", timestamp="2024-01-21T14:30:00Z")

    def test_skips_missing_uuids(self) -> None:
        """UUIDs not found in records_by_uuid are skipped."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
        }
        uuids = ["user1", "missing_uuid"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 1
        assert steps[0] == Step(uuid="user1", timestamp="2024-01-21T14:30:00Z")

    def test_preserves_uuid_order(self) -> None:
        """Steps are returned in the same order as input UUIDs."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "msg_c": {
                "uuid": "msg_c",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "C"},
            },
            "msg_a": {
                "uuid": "msg_a",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "A"}]},
            },
            "msg_b": {
                "uuid": "msg_b",
                "type": "user",
                "timestamp": "2024-01-21T14:30:10Z",
                "message": {"content": "B"},
            },
        }
        uuids = ["msg_c", "msg_a", "msg_b"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 3
        assert steps[0].uuid == "msg_c"
        assert steps[1].uuid == "msg_a"
        assert steps[2].uuid == "msg_b"

    def test_tool_result_becomes_own_step(self) -> None:
        """Tool results (user messages with toolUseResult) are their own steps."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "asst1": {
                "uuid": "asst1",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": [{"type": "tool_use", "name": "Read"}]},
            },
            "tool_result": {
                "uuid": "tool_result",
                "type": "user",
                "timestamp": "2024-01-21T14:30:01Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "file contents here"},
            },
            "asst2": {
                "uuid": "asst2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Done!"}]},
            },
        }
        uuids = ["asst1", "tool_result", "asst2"]

        steps = filter_to_steps(uuids, records_by_uuid)

        # All three become separate steps (tool_result is NOT merged with asst1)
        assert len(steps) == 3
        assert steps[0] == Step(uuid="asst1", timestamp="2024-01-21T14:30:00Z")
        assert steps[1] == Step(uuid="tool_result", timestamp="2024-01-21T14:30:01Z")
        assert steps[2] == Step(uuid="asst2", timestamp="2024-01-21T14:30:05Z")

    def test_empty_uuids_returns_empty_list(self) -> None:
        """Empty UUID list returns empty step list."""
        from ragzoom_claude_code.transcript_sync import filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
        }

        steps = filter_to_steps([], records_by_uuid)

        assert steps == []

    def test_skips_non_string_timestamp(self) -> None:
        """Records with non-string timestamp are skipped."""
        from ragzoom_claude_code.transcript_sync import Step, filter_to_steps

        records_by_uuid: dict[str, dict[str, object]] = {
            "user1": {
                "uuid": "user1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            "user2": {
                "uuid": "user2",
                "type": "user",
                "timestamp": 1234567890,  # Non-string timestamp
                "message": {"content": "Numeric timestamp"},
            },
        }
        uuids = ["user1", "user2"]

        steps = filter_to_steps(uuids, records_by_uuid)

        assert len(steps) == 1
        assert steps[0] == Step(uuid="user1", timestamp="2024-01-21T14:30:00Z")
