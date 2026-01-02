"""Admin CLI for memory service operations.

Usage:
    # Local (with local DB):
    python -m memory_service.admin status

    # On Railway (connects to internal DB):
    railway run python -m memory_service.admin status
    railway run python -m memory_service.admin reset <session_id>

This CLI connects directly to the database, not via gRPC.
Admin operations are intentionally kept out of the customer-facing API.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Iterator

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from memory_service.ingestion.claude.transcript_sync import (
    _build_parent_map_from_bytes,
    _build_records_map_from_bytes,
    _get_current_head_from_bytes,
    get_ancestor_chain,
    transcribe_uuids_from_map,
)
from memory_service.storage import SessionRawData


def get_database_url() -> str | None:
    """Get database URL from environment."""
    return os.environ.get("RAGZOOM_DATABASE_URL") or os.environ.get("DATABASE_URL")


def cmd_status(args: argparse.Namespace) -> int:
    """Show memory service status."""
    db_url = get_database_url()

    print("Memory Service Status")
    print("=" * 50)

    # Database connection
    if not db_url:
        print("\n❌ Database: NOT CONFIGURED")
        print("   Set RAGZOOM_DATABASE_URL or DATABASE_URL")
        return 1

    # Mask password in URL for display
    display_url = db_url
    if "@" in db_url:
        parts = db_url.split("@")
        host_part = parts[-1]
        display_url = f"postgresql://***@{host_part}"
    print(f"\n📊 Database: {display_url}")

    try:
        engine = create_engine(db_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("   ✅ Connected")
    except Exception as e:
        print(f"   ❌ Connection failed: {e}")
        return 1

    # Session inventory
    with Session(engine) as db:
        sessions = db.execute(
            select(
                SessionRawData.session_id,
                SessionRawData.user_id,
                SessionRawData.original_file_offset,
                SessionRawData.span_end,
                SessionRawData.last_synced_uuid,
                func.length(SessionRawData.jsonl_content).label("content_size"),
            ).order_by(SessionRawData.session_id)
        ).all()

        print(f"\n📋 Sessions: {len(sessions)}")
        if sessions:
            print()
            for s in sessions:
                uuid_display = s.last_synced_uuid[:8] if s.last_synced_uuid else "None"
                print(f"   {s.session_id[:12]}...")
                print(f"      user: {s.user_id}")
                print(f"      offset: {s.original_file_offset:,} bytes")
                print(f"      span_end: {s.span_end}")
                print(f"      last_synced: {uuid_display}")
                print(f"      stored: {s.content_size:,} bytes")
                print()

        # Document stats (if ragzoom tables exist)
        try:
            doc_count = db.execute(text("SELECT COUNT(*) FROM documents")).scalar_one()
            node_count = db.execute(
                text("SELECT COUNT(*) FROM tree_nodes")
            ).scalar_one()
            print(f"📄 Documents: {doc_count}")
            print(f"🌳 Tree nodes: {node_count}")
        except Exception:
            print("📄 RagZoom tables: Not found or not accessible")

    return 0


def cmd_reset(args: argparse.Namespace) -> int:
    """Reset a session cursor for full re-sync."""
    session_id = args.session_id
    db_url = get_database_url()

    if not db_url:
        print("❌ Database not configured")
        print("   Set RAGZOOM_DATABASE_URL or DATABASE_URL")
        return 1

    engine = create_engine(db_url)
    with Session(engine) as db:
        # Find the session
        row = db.execute(
            select(SessionRawData).where(SessionRawData.session_id == session_id)
        ).scalar_one_or_none()

        if row is None:
            # Try partial match
            rows = (
                db.execute(
                    select(SessionRawData).where(
                        SessionRawData.session_id.like(f"{session_id}%")
                    )
                )
                .scalars()
                .all()
            )

            if len(rows) == 0:
                print(f"❌ Session not found: {session_id}")
                return 1
            elif len(rows) > 1:
                print(f"❌ Multiple sessions match '{session_id}':")
                for r in rows:
                    print(f"   {r.session_id}")
                return 1
            else:
                row = rows[0]

        print(f"Resetting session: {row.session_id}")
        print(f"   Current offset: {row.original_file_offset:,}")
        print(f"   Current span_end: {row.span_end}")
        print(f"   Current last_synced: {row.last_synced_uuid}")

        # Reset cursor fields
        row.last_synced_uuid = None
        row.original_file_offset = 0
        # Note: span_end stays as-is to trigger revert detection on next sync
        db.commit()

        print()
        print("✅ Cursor reset. Next sync will trigger full re-index.")

    return 0


def _find_session(db: Session, session_id: str) -> SessionRawData | None:
    """Find a session by exact or partial match."""
    row = db.execute(
        select(SessionRawData).where(SessionRawData.session_id == session_id)
    ).scalar_one_or_none()

    if row is not None:
        return row

    # Try partial match
    rows = (
        db.execute(
            select(SessionRawData).where(
                SessionRawData.session_id.like(f"{session_id}%")
            )
        )
        .scalars()
        .all()
    )

    if len(rows) == 0:
        print(f"❌ Session not found: {session_id}")
        return None
    elif len(rows) > 1:
        print(f"❌ Multiple sessions match '{session_id}':")
        for r in rows:
            print(f"   {r.session_id}")
        return None
    else:
        return rows[0]


def _segment_uuids(
    uuids: list[str], records_map: dict[str, dict[str, object]]
) -> list[list[str]]:
    """Split UUIDs into segments at turn boundaries (same as execute_sync)."""
    segments: list[list[str]] = []
    current_segment: list[str] = []
    prev_was_user = False

    for uuid in uuids:
        record = records_map.get(uuid)
        if record is None:
            continue

        is_user_message = record.get("type") == "user" and "toolUseResult" not in record

        if is_user_message and not prev_was_user and current_segment:
            segments.append(current_segment)
            current_segment = []

        current_segment.append(uuid)
        prev_was_user = is_user_message

    if current_segment:
        segments.append(current_segment)

    return segments


def _transcribe_session(content: bytes) -> str:
    """Transcribe stored JSONL content to readable text.

    This matches the actual indexing behavior: segments are transcribed
    individually and concatenated WITHOUT separators.
    """
    parent_map = _build_parent_map_from_bytes(content)
    current_head = _get_current_head_from_bytes(content)
    if current_head is None:
        return ""

    uuids = get_ancestor_chain(current_head, None, parent_map)
    records_map = _build_records_map_from_bytes(content, set(uuids))

    # Split into segments (same as execute_sync)
    segments = _segment_uuids(uuids, records_map)

    # Transcribe each segment and concatenate WITHOUT separators
    # (this matches how batch_append accumulates spans)
    segment_texts: list[str] = []
    for segment_uuids in segments:
        text = transcribe_uuids_from_map(segment_uuids, records_map)
        if text:
            segment_texts.append(text)

    return "".join(segment_texts)


def cmd_transcribe(args: argparse.Namespace) -> int:
    """Transcribe session from stored JSONL."""
    session_id = args.session_id
    output_path = args.output
    db_url = get_database_url()

    if not db_url:
        print("❌ Database not configured", file=sys.stderr)
        return 1

    engine = create_engine(db_url)
    with Session(engine) as db:
        row = _find_session(db, session_id)
        if row is None:
            return 1

        content = row.jsonl_content
        if not content:
            print("❌ Session has no stored content", file=sys.stderr)
            return 1

        text_output = _transcribe_session(content)

        if output_path:
            with open(output_path, "w") as f:
                f.write(text_output)
            print(f"✅ Transcribed to {output_path}", file=sys.stderr)
        else:
            print(text_output)

    return 0


def _iter_transcription_chars(
    content: bytes,
) -> Iterator[tuple[str, str, int]]:
    """Yield (char, current_uuid, char_offset) for streaming validation.

    This matches actual indexing: segments transcribed individually,
    concatenated WITHOUT separators.
    """
    parent_map = _build_parent_map_from_bytes(content)
    current_head = _get_current_head_from_bytes(content)
    if current_head is None:
        return

    uuids = get_ancestor_chain(current_head, None, parent_map)
    records_map = _build_records_map_from_bytes(content, set(uuids))

    # Split into segments and transcribe (matches execute_sync)
    segments = _segment_uuids(uuids, records_map)

    # Build segment boundaries for UUID lookup
    segment_boundaries: list[tuple[int, int, str]] = []
    offset = 0
    for segment_uuids in segments:
        text = transcribe_uuids_from_map(segment_uuids, records_map)
        if text:
            # Use last UUID in segment for error reporting
            last_uuid = segment_uuids[-1]
            segment_boundaries.append((offset, offset + len(text), last_uuid))
            offset += len(text)

    # Get full concatenated text
    full_text = _transcribe_session(content)

    def get_uuid_for_offset(char_offset: int) -> str:
        for start, end, uuid in segment_boundaries:
            if start <= char_offset < end:
                return uuid
        return segment_boundaries[-1][2] if segment_boundaries else ""

    for i, char in enumerate(full_text):
        yield (char, get_uuid_for_offset(i), i)


def _iter_leaf_chars(db: Session, document_id: str) -> Iterator[tuple[str, int]]:
    """Yield (char, span_offset) streaming from leaves."""
    # Stream leaves ordered by span_start
    result = db.execute(
        text(
            """
            SELECT text, span_start FROM tree_nodes
            WHERE document_id = :doc_id AND height = 0
            ORDER BY span_start
            """
        ),
        {"doc_id": document_id},
    )

    for row in result:
        leaf_text = row.text
        span_start = row.span_start
        for i, char in enumerate(leaf_text):
            yield (char, span_start + i)


def _normalize_whitespace_stream(
    char_iter: Iterator[tuple[str, str, int]],
) -> Iterator[tuple[str, str, int]]:
    """Normalize whitespace in character stream.

    Collapses runs of whitespace to single space, preserving non-whitespace.
    """
    in_whitespace = False
    for char, uuid, offset in char_iter:
        if char.isspace():
            if not in_whitespace:
                # Emit single space for start of whitespace run
                yield (" ", uuid, offset)
                in_whitespace = True
            # Skip additional whitespace
        else:
            in_whitespace = False
            yield (char, uuid, offset)


def _normalize_leaf_stream(
    char_iter: Iterator[tuple[str, int]],
) -> Iterator[tuple[str, int]]:
    """Normalize whitespace in leaf character stream."""
    in_whitespace = False
    for char, offset in char_iter:
        if char.isspace():
            if not in_whitespace:
                yield (" ", offset)
                in_whitespace = True
        else:
            in_whitespace = False
            yield (char, offset)


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate indexed leaves match fresh transcription."""
    session_id = args.session_id
    db_url = get_database_url()

    if not db_url:
        print("❌ Database not configured", file=sys.stderr)
        return 1

    engine = create_engine(db_url)
    with Session(engine) as db:
        row = _find_session(db, session_id)
        if row is None:
            return 1

        full_session_id = row.session_id
        content = row.jsonl_content
        if not content:
            print("❌ Session has no stored content", file=sys.stderr)
            return 1

        # Create streaming iterators with whitespace normalization
        trans_iter = _normalize_whitespace_stream(_iter_transcription_chars(content))
        leaf_iter = _normalize_leaf_stream(_iter_leaf_chars(db, full_session_id))

        # Track context for error reporting
        context_buffer: list[str] = []
        context_size = 50

        # Compare character by character
        trans_exhausted = False
        leaf_exhausted = False
        last_uuid = ""
        last_offset = 0
        chars_compared = 0

        while True:
            try:
                trans_char, uuid, trans_offset = next(trans_iter)
                last_uuid = uuid
                last_offset = trans_offset
            except StopIteration:
                trans_exhausted = True
                trans_char = None

            try:
                leaf_char, leaf_offset = next(leaf_iter)
            except StopIteration:
                leaf_exhausted = True
                leaf_char = None

            # Both exhausted - success!
            if trans_exhausted and leaf_exhausted:
                print(f"✅ Validation passed - {chars_compared:,} characters match")
                return 0

            # One exhausted before other - length mismatch
            if trans_exhausted:
                print("❌ Leaf content continues past transcription end")
                print(f"   Transcription ended at offset {last_offset:,}")
                print(f"   Last UUID: {last_uuid}")
                print(f"   Extra leaf content: {repr(leaf_char)}...")
                return 1

            if leaf_exhausted:
                print("❌ Transcription continues past leaf content end")
                print(f"   Leaves ended at offset {leaf_offset:,}")
                print(f"   Current UUID: {uuid}")
                print(f"   Extra transcription: {repr(trans_char)}...")
                return 1

            # At this point neither is exhausted, so chars are not None
            assert trans_char is not None
            assert leaf_char is not None

            chars_compared += 1

            # Track context
            context_buffer.append(trans_char)
            if len(context_buffer) > context_size:
                context_buffer.pop(0)

            # Compare
            if trans_char != leaf_char:
                # Collect some following context
                following: list[str] = [leaf_char]
                for _ in range(context_size):
                    try:
                        c, _ = next(leaf_iter)
                        following.append(c)
                    except StopIteration:
                        break

                print(f"❌ Divergence at offset {trans_offset:,}")
                print(f"   UUID: {uuid}")
                print(f"   Expected: {repr(trans_char)}")
                print(f"   Got:      {repr(leaf_char)}")
                print()
                print(f"   Context before: {''.join(context_buffer[:-1])!r}")
                print(f"   Leaf continues: {''.join(following)!r}")
                return 1

    return 0


def cmd_inspect_uuid(args: argparse.Namespace) -> int:
    """Inspect raw JSONL record for a UUID."""
    session_id = args.session_id
    target_uuid = args.uuid
    context = args.context
    db_url = get_database_url()

    if not db_url:
        print("❌ Database not configured", file=sys.stderr)
        return 1

    engine = create_engine(db_url)
    with Session(engine) as db:
        row = _find_session(db, session_id)
        if row is None:
            return 1

        content = row.jsonl_content
        if not content:
            print("❌ Session has no stored content", file=sys.stderr)
            return 1

        # Build full records map to find the target
        parent_map = _build_parent_map_from_bytes(content)
        current_head = _get_current_head_from_bytes(content)
        if current_head is None:
            print("❌ No current head found", file=sys.stderr)
            return 1

        uuids = get_ancestor_chain(current_head, None, parent_map)

        # Find target UUID position (partial match)
        target_idx = None
        for i, uuid in enumerate(uuids):
            if uuid.startswith(target_uuid) or target_uuid in uuid:
                target_idx = i
                target_uuid = uuid  # Use full UUID
                break

        if target_idx is None:
            print(f"❌ UUID {target_uuid} not found in ancestor chain")
            return 1

        # Get context window
        start_idx = max(0, target_idx - context)
        end_idx = min(len(uuids), target_idx + context + 1)
        context_uuids = uuids[start_idx:end_idx]

        records_map = _build_records_map_from_bytes(content, set(context_uuids))

        print(f"UUID {target_uuid}")
        print(f"Position in chain: {target_idx} / {len(uuids)}")
        print(f"Showing {start_idx} to {end_idx - 1}")
        print("=" * 70)

        for i, uuid in enumerate(context_uuids):
            actual_idx = start_idx + i
            marker = ">>>" if uuid == target_uuid else "   "
            record = records_map.get(uuid, {})

            msg_type = record.get("type", "?")
            parent = record.get("parentUuid", "?")

            # Get content preview
            content_preview = ""
            if msg_type == "user":
                msg = record.get("message", {})
                if isinstance(msg, dict):
                    content_preview = str(msg.get("content", ""))[:100]
            elif msg_type == "assistant":
                msg = record.get("message", {})
                if isinstance(msg, dict):
                    content_list = msg.get("content", [])
                    if isinstance(content_list, list):
                        for item in content_list:
                            if isinstance(item, dict) and item.get("type") == "text":
                                content_preview = str(item.get("text", ""))[:100]
                                break

            print(f"{marker} [{actual_idx}] {uuid[:12]}... type={msg_type}")
            print(f"       parent={str(parent)[:12] if parent else 'None'}...")
            if content_preview:
                print(f"       content: {content_preview!r}")
            print()

    return 0


def cmd_inspect_leaves(args: argparse.Namespace) -> int:
    """Inspect indexed leaves around a span offset."""
    session_id = args.session_id
    offset = args.offset
    context = args.context
    db_url = get_database_url()

    if not db_url:
        print("❌ Database not configured", file=sys.stderr)
        return 1

    engine = create_engine(db_url)
    with Session(engine) as db:
        row = _find_session(db, session_id)
        if row is None:
            return 1

        full_session_id = row.session_id

        # Find leaves around the offset
        result = db.execute(
            text(
                """
                SELECT span_start, span_end, text
                FROM tree_nodes
                WHERE document_id = :doc_id AND height = 0
                AND span_start <= :offset_end AND span_end >= :offset_start
                ORDER BY span_start
                """
            ),
            {
                "doc_id": full_session_id,
                "offset_start": offset - context,
                "offset_end": offset + context,
            },
        ).all()

        print(f"Leaves around offset {offset:,} (±{context:,})")
        print("=" * 70)

        for leaf in result:
            # Highlight if this leaf contains the exact offset
            contains_offset = leaf.span_start <= offset < leaf.span_end
            marker = ">>>" if contains_offset else "   "

            print(f"{marker} [{leaf.span_start:,}-{leaf.span_end:,}]")

            # Show text with offset marker if applicable
            text_to_show = leaf.text
            if len(text_to_show) > 500:
                if contains_offset:
                    # Show around the offset
                    rel_offset = offset - leaf.span_start
                    start = max(0, rel_offset - 100)
                    end = min(len(text_to_show), rel_offset + 100)
                    text_to_show = (
                        ("..." if start > 0 else "")
                        + text_to_show[start:end]
                        + ("..." if end < len(text_to_show) else "")
                    )
                else:
                    text_to_show = text_to_show[:200] + "..."

            print(f"       {text_to_show!r}")
            print()

    return 0


def cmd_chain(args: argparse.Namespace) -> int:
    """Show ancestor chain summary."""
    session_id = args.session_id
    limit = args.limit
    db_url = get_database_url()

    if not db_url:
        print("❌ Database not configured", file=sys.stderr)
        return 1

    engine = create_engine(db_url)
    with Session(engine) as db:
        row = _find_session(db, session_id)
        if row is None:
            return 1

        content = row.jsonl_content
        if not content:
            print("❌ Session has no stored content", file=sys.stderr)
            return 1

        parent_map = _build_parent_map_from_bytes(content)
        current_head = _get_current_head_from_bytes(content)
        if current_head is None:
            print("❌ No current head found", file=sys.stderr)
            return 1

        uuids = get_ancestor_chain(current_head, None, parent_map)
        records_map = _build_records_map_from_bytes(content, set(uuids))

        print(f"Ancestor chain: {len(uuids)} messages")
        print(f"Head: {current_head}")
        print(f"Last synced: {row.last_synced_uuid}")
        print("=" * 70)

        # Show first and last N messages
        show_uuids = uuids[:limit] if limit else uuids
        if limit and len(uuids) > limit * 2:
            show_uuids = uuids[:limit] + ["..."] + uuids[-limit:]

        for i, uuid in enumerate(show_uuids):
            if uuid == "...":
                print(f"   ... ({len(uuids) - limit * 2} more messages) ...")
                continue

            actual_idx = uuids.index(uuid)
            record = records_map.get(uuid, {})
            msg_type = record.get("type", "?")

            # Mark if this is the last synced UUID
            marker = "   "
            if uuid == row.last_synced_uuid:
                marker = ">>>"

            print(f"{marker} [{actual_idx}] {uuid[:12]}... type={msg_type}")

    return 0


def cmd_segments(args: argparse.Namespace) -> int:
    """Show segment boundaries and content summary."""
    session_id = args.session_id
    db_url = get_database_url()

    if not db_url:
        print("❌ Database not configured", file=sys.stderr)
        return 1

    engine = create_engine(db_url)
    with Session(engine) as db:
        row = _find_session(db, session_id)
        if row is None:
            return 1

        content = row.jsonl_content
        if not content:
            print("❌ Session has no stored content", file=sys.stderr)
            return 1

        parent_map = _build_parent_map_from_bytes(content)
        current_head = _get_current_head_from_bytes(content)
        if current_head is None:
            print("❌ No current head found", file=sys.stderr)
            return 1

        uuids = get_ancestor_chain(current_head, None, parent_map)
        records_map = _build_records_map_from_bytes(content, set(uuids))
        segments = _segment_uuids(uuids, records_map)

        print(f"Segments: {len(segments)}")
        print("=" * 70)

        char_offset = 0
        for i, segment_uuids in enumerate(segments):
            text = transcribe_uuids_from_map(segment_uuids, records_map)
            text_len = len(text) if text else 0

            # Get first record type info
            first_record = records_map.get(segment_uuids[0], {})
            first_type = first_record.get("type", "?")

            print(f"Segment {i}: {len(segment_uuids)} msgs, {text_len:,} chars")
            print(f"   Span: {char_offset:,} - {char_offset + text_len:,}")
            print(f"   First UUID: {segment_uuids[0][:12]}... ({first_type})")
            print(f"   Last UUID: {segment_uuids[-1][:12]}...")
            if text:
                preview = text[:100].replace("\n", "\\n")
                print(f"   Preview: {preview!r}")
            print()

            char_offset += text_len

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Memory service admin CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # status command
    subparsers.add_parser("status", help="Show memory service status")

    # reset command
    reset_parser = subparsers.add_parser(
        "reset", help="Reset a session cursor for full re-sync"
    )
    reset_parser.add_argument("session_id", help="Session ID (or prefix) to reset")

    # transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe session from stored JSONL"
    )
    transcribe_parser.add_argument("session_id", help="Session ID (or prefix)")
    transcribe_parser.add_argument(
        "--output", "-o", help="Output file (default: stdout)"
    )

    # validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate indexed leaves match transcription"
    )
    validate_parser.add_argument("session_id", help="Session ID (or prefix)")

    # inspect-uuid command
    inspect_uuid_parser = subparsers.add_parser(
        "inspect-uuid", help="Inspect raw JSONL record for a UUID"
    )
    inspect_uuid_parser.add_argument("session_id", help="Session ID (or prefix)")
    inspect_uuid_parser.add_argument("uuid", help="UUID (or prefix) to inspect")
    inspect_uuid_parser.add_argument(
        "--context", "-c", type=int, default=3, help="Number of messages before/after"
    )

    # inspect-leaves command
    inspect_leaves_parser = subparsers.add_parser(
        "inspect-leaves", help="Inspect indexed leaves around a span offset"
    )
    inspect_leaves_parser.add_argument("session_id", help="Session ID (or prefix)")
    inspect_leaves_parser.add_argument(
        "offset", type=int, help="Span offset to inspect"
    )
    inspect_leaves_parser.add_argument(
        "--context", "-c", type=int, default=1000, help="Bytes of context"
    )

    # chain command
    chain_parser = subparsers.add_parser("chain", help="Show ancestor chain summary")
    chain_parser.add_argument("session_id", help="Session ID (or prefix)")
    chain_parser.add_argument(
        "--limit", "-n", type=int, default=10, help="Messages to show at start/end"
    )

    # segments command
    segments_parser = subparsers.add_parser("segments", help="Show segment boundaries")
    segments_parser.add_argument("session_id", help="Session ID (or prefix)")

    args = parser.parse_args()

    if args.command == "status":
        return cmd_status(args)
    elif args.command == "reset":
        return cmd_reset(args)
    elif args.command == "transcribe":
        return cmd_transcribe(args)
    elif args.command == "validate":
        return cmd_validate(args)
    elif args.command == "inspect-uuid":
        return cmd_inspect_uuid(args)
    elif args.command == "inspect-leaves":
        return cmd_inspect_leaves(args)
    elif args.command == "chain":
        return cmd_chain(args)
    elif args.command == "segments":
        return cmd_segments(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
