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

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session

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

    args = parser.parse_args()

    if args.command == "status":
        return cmd_status(args)
    elif args.command == "reset":
        return cmd_reset(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
