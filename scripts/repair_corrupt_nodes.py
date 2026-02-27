#!/usr/bin/env python3
"""Repair corrupt nodes in a ragzoom document tree.

Fixes two classes of corruption:
1. Leaves with empty text and bogus timestamps (0.0) — sets timestamps to
   interpolated values from neighboring leaves so forest invariants hold.
2. Orphaned children whose parent_id points to non-existent nodes — sets
   parent_id to NULL so the summarizer can rebuild the tree naturally.

Usage:
    python scripts/repair_corrupt_nodes.py <document_id> [--database-url URL] [--dry-run]

Example (production):
    python scripts/repair_corrupt_nodes.py 6c21718f-f095-483f-8cd6-610137d581aa \
        --database-url sqlite:///data/ragzoom/data/sqlite.db
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys

LOGGER = logging.getLogger("repair_corrupt_nodes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair corrupt nodes (empty text, bad timestamps, orphaned parents) in a ragzoom document."
    )
    parser.add_argument("document_id", help="Target document ID")
    parser.add_argument(
        "--database-url",
        type=str,
        required=True,
        help="SQLite database URL (e.g., sqlite:///path/to/sqlite.db) or bare file path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without mutating the database",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def resolve_db_path(url: str) -> str:
    """Extract file path from a SQLite URL or bare path."""
    if url.startswith("sqlite:///"):
        return url[len("sqlite:///") :]
    if url.startswith("sqlite://"):
        raise ValueError(
            "Only file-backed SQLite databases are supported (use sqlite:///path)"
        )
    return url


def find_corrupt_leaves(conn: sqlite3.Connection, document_id: str) -> list[dict]:
    """Find leaves with empty text (the primary corruption signal)."""
    cursor = conn.execute(
        """
        SELECT id, text, time_start, time_end, span_start, span_end,
               height, level_index, parent_id,
               preceding_neighbor_id, following_neighbor_id
        FROM tree_nodes
        WHERE document_id = ? AND text = '' AND height = 0
        ORDER BY level_index
        """,
        (document_id,),
    )
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def find_nodes_with_zero_timestamps(
    conn: sqlite3.Connection, document_id: str
) -> list[dict]:
    """Find all nodes (any height) with time_start=0.0 or time_end=0.0."""
    cursor = conn.execute(
        """
        SELECT id, text, time_start, time_end, span_start, span_end,
               height, level_index, parent_id,
               preceding_neighbor_id, following_neighbor_id
        FROM tree_nodes
        WHERE document_id = ?
          AND ((time_start IS NOT NULL AND time_start = 0.0)
               OR (time_end IS NOT NULL AND time_end = 0.0))
        ORDER BY height, level_index
        """,
        (document_id,),
    )
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def find_orphaned_children(conn: sqlite3.Connection, document_id: str) -> list[dict]:
    """Find nodes whose parent_id points to a non-existent node."""
    cursor = conn.execute(
        """
        SELECT c.id, c.height, c.level_index, c.parent_id, c.span_start, c.span_end
        FROM tree_nodes c
        WHERE c.document_id = ?
          AND c.parent_id IS NOT NULL
          AND c.parent_id NOT IN (
              SELECT id FROM tree_nodes WHERE document_id = ?
          )
        ORDER BY c.height, c.level_index
        """,
        (document_id, document_id),
    )
    columns = [desc[0] for desc in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def get_node_by_id(conn: sqlite3.Connection, node_id: str) -> dict | None:
    """Fetch a single node by ID."""
    cursor = conn.execute(
        """
        SELECT id, text, time_start, time_end, span_start, span_end,
               height, level_index, parent_id,
               preceding_neighbor_id, following_neighbor_id
        FROM tree_nodes
        WHERE id = ?
        """,
        (node_id,),
    )
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [desc[0] for desc in cursor.description]
    return dict(zip(columns, row))


def interpolate_timestamp(
    conn: sqlite3.Connection, node: dict, max_hops: int = 20
) -> tuple[float | None, float | None]:
    """Determine appropriate timestamps for a corrupt node from its neighbors.

    Walks the linked list in both directions until it finds a neighbor with
    valid (non-zero) timestamps. This handles chains of adjacent corrupt nodes.
    """
    time_start = None
    time_end = None

    # Walk backwards to find a valid time_end
    current_id = node.get("preceding_neighbor_id")
    for _ in range(max_hops):
        if not current_id:
            break
        prev = get_node_by_id(conn, current_id)
        if not prev:
            break
        if prev["time_end"] and prev["time_end"] != 0.0:
            time_start = prev["time_end"]
            break
        current_id = prev.get("preceding_neighbor_id")

    # Walk forwards to find a valid time_start
    current_id = node.get("following_neighbor_id")
    for _ in range(max_hops):
        if not current_id:
            break
        nxt = get_node_by_id(conn, current_id)
        if not nxt:
            break
        if nxt["time_start"] and nxt["time_start"] != 0.0:
            time_end = nxt["time_start"]
            break
        current_id = nxt.get("following_neighbor_id")

    # Fall back if one side is missing
    if time_start is None and time_end is not None:
        time_start = time_end
    if time_end is None and time_start is not None:
        time_end = time_start

    return time_start, time_end


def repair_timestamps(conn: sqlite3.Connection, document_id: str, dry_run: bool) -> int:
    """Fix timestamps on all nodes with time_start=0.0 or time_end=0.0.

    For leaves: interpolate from neighboring leaves in the linked list.
    For internal nodes: recompute from children (left child's time_start,
    right child's time_end).
    """
    zero_ts_nodes = find_nodes_with_zero_timestamps(conn, document_id)
    if not zero_ts_nodes:
        LOGGER.info("No nodes with zero timestamps found.")
        return 0

    # Process leaves first (height 0), then internal nodes bottom-up
    zero_ts_nodes.sort(key=lambda n: (n["height"], n["level_index"]))

    fixed = 0
    for node in zero_ts_nodes:
        if node["height"] == 0:
            new_start, new_end = interpolate_timestamp(conn, node)
        else:
            # Internal node: recompute from children after they've been fixed
            new_start, new_end = recompute_internal_timestamps(conn, node)

        if new_start is None and new_end is None:
            LOGGER.warning(
                "  Cannot determine timestamps for node %s (h=%d, li=%d) — no valid neighbors",
                node["id"][:12],
                node["height"],
                node["level_index"],
            )
            continue

        changes = []
        if node["time_start"] == 0.0 and new_start is not None:
            changes.append(f"time_start: 0.0 → {new_start}")
        if node["time_end"] == 0.0 and new_end is not None:
            changes.append(f"time_end: 0.0 → {new_end}")

        if not changes:
            continue

        LOGGER.info(
            "  Fix timestamps on %s (h=%d, li=%d): %s",
            node["id"][:12],
            node["height"],
            node["level_index"],
            ", ".join(changes),
        )

        updates = {}
        if node["time_start"] == 0.0 and new_start is not None:
            updates["time_start"] = new_start
        if node["time_end"] == 0.0 and new_end is not None:
            updates["time_end"] = new_end

        if not updates:
            continue

        fixed += 1
        if not dry_run:
            set_clause = ", ".join(f"{k} = ?" for k in updates)
            values = list(updates.values()) + [node["id"]]
            conn.execute(
                f"UPDATE tree_nodes SET {set_clause} WHERE id = ?",
                values,
            )

    return fixed


def recompute_internal_timestamps(
    conn: sqlite3.Connection, node: dict
) -> tuple[float | None, float | None]:
    """Recompute an internal node's timestamps from its children.

    Parent time_start = left child's time_start
    Parent time_end = right child's time_end
    """
    cursor = conn.execute(
        """
        SELECT id, time_start, time_end
        FROM tree_nodes
        WHERE parent_id = ?
        ORDER BY span_start
        """,
        (node["id"],),
    )
    children = cursor.fetchall()

    if not children:
        # Node has no children pointing to it — try the linked list like a leaf
        return interpolate_timestamp(conn, node)

    # Left child = first by span, right child = last by span
    left_start = children[0][1]  # time_start of leftmost child
    right_end = children[-1][2]  # time_end of rightmost child

    time_start = left_start if left_start and left_start != 0.0 else None
    time_end = right_end if right_end and right_end != 0.0 else None

    # Fall back to neighbor interpolation if children also have bad timestamps
    if time_start is None or time_end is None:
        neighbor_start, neighbor_end = interpolate_timestamp(conn, node)
        if time_start is None:
            time_start = neighbor_start
        if time_end is None:
            time_end = neighbor_end

    return time_start, time_end


def repair_orphaned_parents(
    conn: sqlite3.Connection, document_id: str, dry_run: bool
) -> int:
    """NULL out parent_id on nodes whose parent doesn't exist."""
    orphans = find_orphaned_children(conn, document_id)
    if not orphans:
        LOGGER.info("No orphaned children found.")
        return 0

    for orphan in orphans:
        LOGGER.info(
            "  NULL parent_id on %s (h=%d, li=%d, was parent=%s)",
            orphan["id"][:12],
            orphan["height"],
            orphan["level_index"],
            orphan["parent_id"][:12],
        )

    if not dry_run:
        orphan_ids = [o["id"] for o in orphans]
        placeholders = ", ".join("?" for _ in orphan_ids)
        conn.execute(
            f"UPDATE tree_nodes SET parent_id = NULL WHERE id IN ({placeholders})",
            orphan_ids,
        )

    return len(orphans)


def print_summary(conn: sqlite3.Connection, document_id: str) -> None:
    """Print document health summary."""
    cursor = conn.execute(
        """
        SELECT
            COUNT(*) as total,
            COUNT(*) FILTER (WHERE height = 0) as leaves,
            COUNT(*) FILTER (WHERE text = '') as empty_text,
            COUNT(*) FILTER (WHERE time_start = 0.0 OR time_end = 0.0) as zero_ts,
            COUNT(*) FILTER (WHERE parent_id IS NOT NULL AND parent_id NOT IN (
                SELECT id FROM tree_nodes WHERE document_id = ?
            )) as orphaned
        FROM tree_nodes
        WHERE document_id = ?
        """,
        (document_id, document_id),
    )
    row = cursor.fetchone()
    LOGGER.info(
        "Document health: total=%d, leaves=%d, empty_text=%d, zero_timestamps=%d, orphaned=%d",
        row[0],
        row[1],
        row[2],
        row[3],
        row[4],
    )


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    db_path = resolve_db_path(args.database_url)
    LOGGER.info("Connecting to %s", db_path)

    conn = sqlite3.connect(db_path)
    try:
        # Verify document exists
        cursor = conn.execute(
            "SELECT COUNT(*) FROM tree_nodes WHERE document_id = ?",
            (args.document_id,),
        )
        count = cursor.fetchone()[0]
        if count == 0:
            LOGGER.error("No nodes found for document %s", args.document_id)
            sys.exit(1)
        LOGGER.info("Document %s has %d nodes", args.document_id, count)

        LOGGER.info("\n=== Pre-repair health ===")
        print_summary(conn, args.document_id)

        LOGGER.info(
            "\n=== Step 1: Fix timestamps (0.0 → interpolated from neighbors) ==="
        )
        ts_fixed = repair_timestamps(conn, args.document_id, args.dry_run)

        LOGGER.info("\n=== Step 2: NULL orphaned parent_id references ===")
        orphans_fixed = repair_orphaned_parents(conn, args.document_id, args.dry_run)

        if not args.dry_run:
            conn.commit()
            LOGGER.info("\n=== Post-repair health ===")
            print_summary(conn, args.document_id)

        action = "Would fix" if args.dry_run else "Fixed"
        LOGGER.info(
            "\n%s %d timestamp(s) and %d orphaned parent reference(s).",
            action,
            ts_fixed,
            orphans_fixed,
        )

    finally:
        conn.close()


if __name__ == "__main__":
    main()
