"""Tests verifying transcript sync uses claude-transcriber library."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from ragzoom_claude_code.transcript_sync import (
    Step,
    steps_to_append_units,
    transcribe_uuids_from_map,
)


def _records(*items: dict[str, object]) -> dict[str, dict[str, object]]:
    """Helper to build records_by_uuid from a list of records."""
    result: dict[str, dict[str, object]] = {}
    for item in items:
        uuid = item.get("uuid")
        if isinstance(uuid, str):
            result[uuid] = item
    return result


class TestUsesClaudeTranscriber:
    """Verify transcript sync uses claude-transcriber for transcription."""

    def test_transcribe_uuids_from_map_uses_transcriber(self) -> None:
        """transcribe_uuids_from_map delegates to claude-transcriber.Transcriber."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi there!"}]},
            },
        )

        # Patch the Transcriber class to verify it's being used
        with patch(
            "ragzoom_claude_code.transcript_sync.Transcriber"
        ) as mock_transcriber_cls:
            mock_instance = MagicMock()
            mock_instance.transcribe.side_effect = ["User text", "Assistant text"]
            mock_transcriber_cls.return_value = mock_instance

            result = transcribe_uuids_from_map(["msg1", "msg2"], records)

            # Verify Transcriber was instantiated and used
            mock_transcriber_cls.assert_called_once()
            assert mock_instance.transcribe.call_count == 2

            # Verify the result uses transcriber output
            assert "User text" in result
            assert "Assistant text" in result

    def test_steps_to_append_units_uses_transcriber(self) -> None:
        """steps_to_append_units uses claude-transcriber via transcribe_uuids_from_map."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi there!"}]},
            },
        )
        steps = [
            Step(uuid="msg1", timestamp="2024-01-21T14:30:00Z"),
            Step(uuid="msg2", timestamp="2024-01-21T14:30:05Z"),
        ]

        with patch(
            "ragzoom_claude_code.transcript_sync.Transcriber"
        ) as mock_transcriber_cls:
            mock_instance = MagicMock()
            mock_instance.transcribe.side_effect = ["User text", "Assistant text"]
            mock_transcriber_cls.return_value = mock_instance

            result = steps_to_append_units(steps, records)

            # Verify Transcriber was used for each step's text
            mock_transcriber_cls.assert_called()
            assert len(result) == 2  # Each step becomes its own AppendUnit
            # Each step should contain its transcribed content
            assert "User text" in result[0].text
            assert "Assistant text" in result[1].text

    def test_transcriber_preserves_record_order(self) -> None:
        """Records are transcribed in order for proper tool result matching."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:02Z",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "x"},
                        }
                    ]
                },
            },
            {
                "uuid": "msg3",
                "type": "user",
                "timestamp": "2024-01-21T14:30:03Z",
                "toolUseResult": {"type": "success"},
                "message": {"content": "file content"},
            },
            {
                "uuid": "msg4",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Done"}]},
            },
        )

        transcribed_records: list[dict[str, object]] = []

        with patch(
            "ragzoom_claude_code.transcript_sync.Transcriber"
        ) as mock_transcriber_cls:
            mock_instance = MagicMock()

            def capture_and_return(record: dict[str, object]) -> str:
                transcribed_records.append(record)
                return f"Record {len(transcribed_records)}"

            mock_instance.transcribe.side_effect = capture_and_return
            mock_transcriber_cls.return_value = mock_instance

            transcribe_uuids_from_map(["msg1", "msg2", "msg3", "msg4"], records)

            # Verify records were passed to transcriber in order
            assert len(transcribed_records) == 4
            assert transcribed_records[0].get("uuid") == "msg1"
            assert transcribed_records[1].get("uuid") == "msg2"
            assert transcribed_records[2].get("uuid") == "msg3"
            assert transcribed_records[3].get("uuid") == "msg4"

    def test_transcriber_single_instance_per_batch(self) -> None:
        """A single Transcriber instance is used per batch for stateful tool tracking."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi!"}]},
            },
        )

        with patch(
            "ragzoom_claude_code.transcript_sync.Transcriber"
        ) as mock_transcriber_cls:
            mock_instance = MagicMock()
            mock_instance.transcribe.return_value = "text"
            mock_transcriber_cls.return_value = mock_instance

            transcribe_uuids_from_map(["msg1", "msg2"], records)

            # Only one Transcriber instance created (stateful for tool tracking)
            assert mock_transcriber_cls.call_count == 1

    def test_transcriber_skips_none_results(self) -> None:
        """Records that return None from transcriber are skipped."""
        records = _records(
            {
                "uuid": "msg1",
                "type": "user",
                "timestamp": "2024-01-21T14:30:00Z",
                "message": {"content": "Hello"},
            },
            {
                "uuid": "msg2",
                "type": "assistant",
                "timestamp": "2024-01-21T14:30:05Z",
                "message": {"content": [{"type": "text", "text": "Hi!"}]},
            },
        )

        with patch(
            "ragzoom_claude_code.transcript_sync.Transcriber"
        ) as mock_transcriber_cls:
            mock_instance = MagicMock()
            # First record returns None (skipped), second returns text
            mock_instance.transcribe.side_effect = [None, "Assistant response"]
            mock_transcriber_cls.return_value = mock_instance

            result = transcribe_uuids_from_map(["msg1", "msg2"], records)

            # Only the non-None result should be in output
            assert result == "Assistant response"
            assert "None" not in result
