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
from typing import TYPE_CHECKING

# Admin tools connect to explicit database URLs without worktree isolation
os.environ.setdefault("RAGZOOM_SKIP_WORKTREE_ISOLATION", "1")
from collections.abc import Iterator

if TYPE_CHECKING:
    from ragzoom.contracts.storage_backend import StorageBackend

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

from memory_service.ingestion.claude.transcript_sync import (
    _build_parent_map_from_bytes,
    _build_records_map_from_bytes,
    _get_current_head_from_bytes,
    get_ancestor_chain,
    transcribe_uuids_from_map,
)
from memory_service.storage import SessionRawData, SessionStorage


def get_database_url() -> str | None:
    """Get database URL from environment."""
    return os.environ.get("RAGZOOM_DATABASE_URL") or os.environ.get("DATABASE_URL")


def cmd_status(args: argparse.Namespace) -> int:
    """Show memory service status - comprehensive dashboard."""
    db_url = get_database_url()

    print("Memory Service Status")
    print("=" * 60)

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
            print(f"🌳 Tree nodes: {node_count:,}")

            # Per-document detailed status
            if doc_count > 0:
                full_validation = getattr(args, "full_validation", False)
                _print_document_status(db, full_validation=full_validation)

        except Exception:
            print("📄 RagZoom tables: Not found or not accessible")

    return 0


def _print_document_status(db: Session, *, full_validation: bool = False) -> None:
    """Print detailed status for each document."""
    from ragzoom.config import OperationalConfig
    from ragzoom.store import create_store

    # Create store once for all validations (avoid repeated initialization)
    db_url = get_database_url()
    store: StorageBackend | None = None
    if db_url:
        try:
            config = OperationalConfig(database_url=db_url)
            store = create_store(config)
        except Exception:
            pass  # Will show validation error per-document

    # Get comprehensive stats per document
    progress_query = text(
        """
        SELECT
            d.id,
            COUNT(*) FILTER (WHERE t.height = 0) as leaf_count,
            COUNT(*) FILTER (WHERE t.height = 0 AND t.embedding IS NOT NULL)
                as embedded_count,
            COUNT(*) FILTER (WHERE t.height > 0) as summary_count,
            COUNT(*) FILTER (WHERE t.parent_id IS NULL) as root_count,
            COUNT(*) FILTER (WHERE t.parent_id IS NULL AND t.height > 0)
                as parentless_internal,
            MAX(t.height) as max_height,
            COUNT(*) as total_nodes
        FROM documents d
        LEFT JOIN tree_nodes t ON t.document_id = d.id
        GROUP BY d.id
        ORDER BY d.id
        """
    )

    for row in db.execute(progress_query):
        doc_id = row.id
        leaf_count = row.leaf_count or 0
        embedded_count = row.embedded_count or 0
        summary_count = row.summary_count or 0
        root_count = row.root_count or 0
        max_height = row.max_height or 0
        total_nodes = row.total_nodes or 0

        # Calculate expected summaries for a complete forest
        # For a perfect binary tree: summaries = leaves - 1
        # For a forest of N trees: summaries = leaves - N
        # During indexing, use current root_count as approximation
        expected_summaries = max(0, leaf_count - root_count) if leaf_count > 0 else 0

        # Calculate progress percentages
        embed_pct = (embedded_count / leaf_count * 100) if leaf_count > 0 else 0
        summary_pct = (
            (summary_count / expected_summaries * 100) if expected_summaries > 0 else 0
        )

        pending_embeds = leaf_count - embedded_count
        pending_summaries = max(0, expected_summaries - summary_count)

        # Get root distribution by height for job queue analysis
        root_heights = _get_root_height_distribution(db, doc_id)
        mergeable_pairs = sum(count // 2 for count in root_heights.values())

        # Display document ID (truncated for session IDs)
        display_id = doc_id[:40] + "..." if len(doc_id) > 43 else doc_id
        print(f"\n{'─' * 60}")
        print(f"📄 Document: {display_id}")
        print()

        # Indexing Progress
        print("   📈 Indexing Progress:")
        print(f"      Leaves: {leaf_count:,}")

        # Embeddings line
        embed_status = "✅" if pending_embeds == 0 else "⏳"
        print(
            f"      Embeddings: {embedded_count:,}/{leaf_count:,} ({embed_pct:.1f}%) "
            f"{embed_status} {pending_embeds:,} pending"
            if pending_embeds > 0
            else f"      Embeddings: {embedded_count:,}/{leaf_count:,} ({embed_pct:.1f}%) ✅"
        )

        # Summaries line
        summary_status = "✅" if pending_summaries == 0 else "⏳"
        print(
            f"      Summaries: {summary_count:,}/{expected_summaries:,} ({summary_pct:.1f}%) "
            f"{summary_status} ~{pending_summaries:,} pending"
            if pending_summaries > 0
            else f"      Summaries: {summary_count:,}/{expected_summaries:,} ({summary_pct:.1f}%) ✅"
        )

        # Tree structure - forest is complete when no mergeable pairs remain
        # (i.e., no two roots at the same height)
        if mergeable_pairs == 0:
            tree_status = f"✅ Complete forest ({root_count} trees)"
        else:
            tree_status = f"🌲 {root_count} roots, {mergeable_pairs} mergeable pairs"
        print(f"      Tree: height={max_height} | {tree_status}")

        # Validation
        print()
        _print_validation_status(
            db, doc_id, total_nodes, leaf_count, store, full_validation=full_validation
        )


def _get_root_height_distribution(db: Session, document_id: str) -> dict[int, int]:
    """Get count of roots at each height level."""
    result = db.execute(
        text(
            """
            SELECT height, COUNT(*) as count
            FROM tree_nodes
            WHERE document_id = :doc_id AND parent_id IS NULL
            GROUP BY height
            ORDER BY height
            """
        ),
        {"doc_id": document_id},
    )
    return {int(row[0]): int(row[1]) for row in result}


def _validate_transcript(
    db: Session, document_id: str, start_span: int = 0
) -> tuple[bool, str]:
    """Validate indexed leaves match transcript.

    Returns (passed, message) tuple.
    """
    # Get session data
    session = db.execute(
        select(SessionRawData).where(SessionRawData.session_id == document_id)
    ).scalar_one_or_none()

    if session is None:
        return True, "No session data (standalone document)"

    content = session.jsonl_content
    if not content:
        return True, "No stored content"

    # Get full transcript
    full_text = _transcribe_session(bytes(content))
    if not full_text:
        return True, "Empty transcript"

    # Get leaves to validate
    result = db.execute(
        text(
            """
            SELECT span_start, text FROM tree_nodes
            WHERE document_id = :doc_id AND height = 0
            AND span_start >= :start_span
            ORDER BY span_start
            """
        ),
        {"doc_id": document_id, "start_span": start_span},
    ).fetchall()

    if not result:
        return True, "No leaves to validate"

    # Content-based validation: find each leaf in order
    search_pos = 0
    for i, leaf_row in enumerate(result):
        leaf_text = leaf_row.text
        found_pos = full_text.find(leaf_text, search_pos)

        if found_pos < 0:
            # Try normalized search
            normalized_leaf = " ".join(leaf_text.split())
            if len(normalized_leaf) > 20:
                norm_search = " ".join(full_text[search_pos:].split())
                if normalized_leaf[:50] not in norm_search:
                    return (
                        False,
                        f"Leaf {i} (span {leaf_row.span_start:,}) not in transcript",
                    )
        else:
            search_pos = found_pos + len(leaf_text)

    return True, f"{len(result):,} leaves verified"


def _print_validation_status(
    db: Session,
    document_id: str,
    total_nodes: int,
    leaf_count: int,
    store: StorageBackend | None,
    *,
    full_validation: bool = False,
) -> None:
    """Run validation and print results using ragzoom.validation.tree."""
    from ragzoom.validation.tree import validate_document

    if store is None:
        print("   🔍 Validation: ❌ No database URL configured")
        return

    # Run validation (without require_complete since indexing may be in progress)
    # Use fast=True by default for quick status checks; --full-validation disables fast mode
    report = validate_document(
        document_id=document_id,
        store=store,
        require_complete=False,
        fast=not full_validation,
    )

    # Transcript validation (memory-service specific, not in ragzoom.validation)
    transcript_passed, transcript_msg = _validate_transcript(db, document_id)

    # Collect errors and warnings
    errors: list[str] = []
    warnings: list[str] = []

    # Add tree validation errors (limit to first 5 for readability)
    for finding in report.errors[:5]:
        # Summarize duplicate coordinate errors
        if finding.code == "level_neighbors.duplicate_level_index":
            # Extract height and level_index from message
            errors.append(finding.message[:80])
        else:
            errors.append(f"{finding.code}: {finding.message[:60]}")

    if len(report.errors) > 5:
        errors.append(f"... and {len(report.errors) - 5} more errors")

    # Parentless internal nodes shown as info (normal during indexing)
    parentless_count = report.metrics.get("parentless_count", 0)
    if parentless_count > 0:
        warnings.append(f"Internal roots (normal during indexing): {parentless_count}")

    if not transcript_passed:
        errors.append(f"Transcript: {transcript_msg}")

    # Print validation result
    print("   🔍 Validation:")
    if not errors:
        print(
            f"      ✅ Tree: PASSED | Nodes: {total_nodes:,} | Leaves: {leaf_count:,}"
        )
        print(f"      ✅ Transcript: {transcript_msg}")
    else:
        print(f"      ❌ FAILED | Nodes: {total_nodes:,} | Leaves: {leaf_count:,}")
        for error in errors:
            print(f"         • {error}")
    for warning in warnings:
        print(f"         ℹ️  {warning}")


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

        # Use SessionStorage.reset_cursor() to ensure consistent behavior
        # This clears last_synced_uuid, original_file_offset, AND append entries
        storage = SessionStorage(db, user_id=row.user_id)
        storage.reset_cursor(row.session_id)
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


def _transcribe_session(content: bytes | memoryview) -> str:
    """Transcribe stored JSONL content to readable text.

    This matches the actual indexing behavior: segments are transcribed
    individually and concatenated WITHOUT separators.

    Args:
        content: JSONL bytes or memoryview (SQLAlchemy returns memoryview for
            LargeBinary columns).
    """
    # SQLAlchemy returns memoryview for LargeBinary; convert to bytes for .find()
    if isinstance(content, memoryview):
        content = bytes(content)
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
    start_offset: int = 0,
) -> Iterator[tuple[str, str, int]]:
    """Yield (char, current_uuid, char_offset) for streaming validation.

    This matches actual indexing: segments transcribed individually,
    concatenated WITHOUT separators.

    Args:
        content: Raw JSONL bytes
        start_offset: Start yielding from this offset (default 0)
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
        if i >= start_offset:
            yield (char, get_uuid_for_offset(i), i)


def _iter_leaf_chars(
    db: Session, document_id: str, start_span: int = 0
) -> Iterator[tuple[str, int]]:
    """Yield (char, span_offset) streaming from leaves.

    Args:
        db: Database session
        document_id: Document ID to read leaves from
        start_span: Start reading from this span offset (default 0)
    """
    # Stream leaves ordered by span_start, starting from start_span
    result = db.execute(
        text(
            """
            SELECT text, span_start FROM tree_nodes
            WHERE document_id = :doc_id AND height = 0
            AND span_end > :start_span
            ORDER BY span_start
            """
        ),
        {"doc_id": document_id, "start_span": start_span},
    )

    for row in result:
        leaf_text = row.text
        span_start = row.span_start

        # If this leaf starts before start_span, skip the early characters
        char_offset = max(0, start_span - span_start)

        for i, char in enumerate(leaf_text[char_offset:], start=char_offset):
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
    """Validate indexed leaves match fresh transcription.

    Uses content-based validation: verifies each leaf's text appears in the
    transcript in order, handling span drift from incremental indexing.
    """
    session_id = args.session_id
    from_compaction = getattr(args, "from_compaction", False)
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

        # Check for compaction boundary
        compaction_boundary = row.compaction_span_end
        if compaction_boundary:
            print(f"ℹ️  Compaction boundary at span {compaction_boundary:,}")
            if not from_compaction:
                print(
                    "   Use --from-compaction to validate only post-compaction content"
                )
                print()

        # Determine start offset for validation
        start_span = (
            compaction_boundary if from_compaction and compaction_boundary else 0
        )

        # Get full transcript text
        full_text = _transcribe_session(bytes(content))
        print(f"📝 Transcript length: {len(full_text):,} chars")

        # Get leaves to validate, ordered by span
        result = db.execute(
            text(
                """
                SELECT span_start, span_end, text FROM tree_nodes
                WHERE document_id = :doc_id AND height = 0
                AND span_start >= :start_span
                ORDER BY span_start
                """
            ),
            {"doc_id": full_session_id, "start_span": start_span},
        ).fetchall()

        print(f"🌿 Leaves to validate: {len(result)}")
        print()

        # Content-based validation: find each leaf in order in the transcript
        search_pos = 0
        validated_chars = 0
        max_gap = 0

        for i, leaf_row in enumerate(result):
            leaf_text = leaf_row.text
            span_start = leaf_row.span_start

            # Find this leaf's content in the transcript (starting from last pos)
            found_pos = full_text.find(leaf_text, search_pos)

            if found_pos < 0:
                # Try with whitespace normalization
                normalized_leaf = " ".join(leaf_text.split())
                normalized_transcript = " ".join(full_text[search_pos:].split())
                norm_pos = normalized_transcript.find(normalized_leaf[:100])

                if norm_pos < 0:
                    print(f"❌ Leaf {i} (span {span_start:,}) not found in transcript")
                    print(f"   Search started at position {search_pos:,}")
                    print(f"   Leaf content: {repr(leaf_text[:100])}...")
                    print()
                    print(
                        f"   Transcript around search pos: {repr(full_text[search_pos:search_pos+200])}..."
                    )
                    return 1

                # Found with normalization - this is acceptable
                print(f"⚠️  Leaf {i} found with whitespace normalization")

            else:
                gap = found_pos - search_pos
                if gap > max_gap:
                    max_gap = gap
                search_pos = found_pos + len(leaf_text)
                validated_chars += len(leaf_text)

            # Progress indicator every 100 leaves
            if (i + 1) % 100 == 0:
                print(f"   Validated {i + 1}/{len(result)} leaves...")

        print()
        print("✅ Validation passed")
        print(f"   {validated_chars:,} chars in {len(result)} leaves validated")
        print(f"   Max gap between leaves: {max_gap:,} chars")

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
    status_parser = subparsers.add_parser("status", help="Show memory service status")
    status_parser.add_argument(
        "--full-validation",
        action="store_true",
        help="Run full validation (slower, loads all nodes into memory)",
    )

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
    validate_parser.add_argument(
        "--from-compaction",
        action="store_true",
        dest="from_compaction",
        help="Start validation from the compaction boundary (skip pre-compaction content)",
    )

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
