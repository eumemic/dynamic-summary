#!/usr/bin/env python3
"""Turn a fully summarized document tree into a forest by removing random roots."""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency

    def load_dotenv() -> None:  # type: ignore[override]
        return None


# ruff: noqa: E402
# Ensure project root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sqlalchemy import text

from ragzoom.config import OperationalConfig
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.store import create_store
from ragzoom.vector_factory import create_vector_index

LOGGER = logging.getLogger("prune_roots")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly remove non-leaf roots from a document to simulate a partially summarized forest."
    )
    parser.add_argument("document_id", help="Target document ID")
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Number of non-leaf roots to remove (default: 1)",
    )
    parser.add_argument(
        "--database-url",
        type=str,
        help="Override database URL (defaults to OperationalConfig resolution)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sqlite", "postgres"],
        help="Override storage backend",
    )
    parser.add_argument(
        "--vector-backend",
        type=str,
        choices=["python", "chroma", "pgvector"],
        help="Override vector backend",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        help="Explicit embedding model for vector index access",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for reproducible root selection",
    )
    parser.add_argument(
        "--skip-vector",
        action="store_true",
        help="Skip vector index cleanup (use if index is external or unavailable)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which roots would be removed without mutating the database",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> OperationalConfig:
    kwargs = {}
    if args.database_url:
        kwargs["database_url"] = args.database_url
    if args.backend:
        kwargs["backend"] = args.backend
    if args.vector_backend:
        kwargs["vector_backend"] = args.vector_backend
    return OperationalConfig(**kwargs)


def ensure_document(store: DocumentStore, document_id: str) -> None:
    if store.document_id != document_id:
        raise RuntimeError("DocumentStore scoped to unexpected document")
    if store.get_metadata() is None:
        raise RuntimeError(f"Document '{document_id}' does not exist")


def list_non_leaf_roots(store: DocumentStore) -> list[TreeNode]:
    roots = store.nodes.get_root_nodes()
    candidates: list[TreeNode] = []
    for node in roots:
        left_id = getattr(node, "left_child_id", None)
        right_id = getattr(node, "right_child_id", None)
        if left_id or right_id:
            candidates.append(node)
    return candidates


def detach_children(store: DocumentStore, root: TreeNode, session) -> None:
    updates = []
    if root.left_child_id:
        updates.append((root.left_child_id, None))
    if root.right_child_id:
        updates.append((root.right_child_id, None))
    if updates:
        store.nodes.update_parent_references_batch(updates, session=session)


def delete_root(store: DocumentStore, root: TreeNode, session) -> None:
    session.execute(
        text("DELETE FROM tree_nodes WHERE id = :node_id"), {"node_id": root.id}
    )
    cache_manager = getattr(store._node_repo, "cache_manager", None)
    if cache_manager is not None:
        try:
            cache_manager.invalidate(root.id)
        except Exception:  # pragma: no cover - cache is best-effort
            LOGGER.debug("Failed to invalidate cache for %s", root.id, exc_info=True)


def cleanup_neighbors(store: DocumentStore, root: TreeNode, session) -> None:
    # Redirect neighbors that referenced the removed root
    bindings = {
        "root_id": root.id,
        "preceding": root.preceding_neighbor_id or None,
        "following": root.following_neighbor_id or None,
    }
    session.execute(
        text(
            "UPDATE tree_nodes SET preceding_neighbor_id = :preceding WHERE preceding_neighbor_id = :root_id"
        ),
        bindings,
    )
    session.execute(
        text(
            "UPDATE tree_nodes SET following_neighbor_id = :following WHERE following_neighbor_id = :root_id"
        ),
        bindings,
    )


def remove_single_root(
    store: DocumentStore,
    vector_index,
    root: TreeNode,
) -> str:
    LOGGER.info(
        "Removing root %s (height=%s, span=[%s,%s])",
        root.id,
        getattr(root, "height", "?"),
        root.span_start,
        root.span_end,
    )

    with store.transaction() as session:
        detach_children(store, root, session)
        cleanup_neighbors(store, root, session)
        delete_root(store, root, session)

    affected = [root.id]
    if root.left_child_id:
        affected.append(root.left_child_id)
    if root.right_child_id:
        affected.append(root.right_child_id)
    store.tree.clear_depth_cache(affected)

    if vector_index is not None:
        try:
            removed = vector_index.delete(ids=[root.id])
            LOGGER.debug("Vector delete for %s removed %d entries", root.id, removed)
        except Exception as exc:  # pragma: no cover - vector cleanup best effort
            LOGGER.warning("Failed to delete vector for %s: %s", root.id, exc)

    return root.id


def main() -> None:
    load_dotenv()

    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    if args.count <= 0:
        LOGGER.error("--count must be positive")
        sys.exit(1)

    if args.seed is not None:
        random.seed(args.seed)

    config = resolve_config(args)
    store_backend = create_store(config)
    doc_store = store_backend.for_document(args.document_id)

    try:
        ensure_document(doc_store, args.document_id)
    except Exception as exc:
        LOGGER.error("Document '%s' not found: %s", args.document_id, exc)
        sys.exit(1)

    embedding_model = (
        args.embedding_model
        or doc_store.get_embedding_model()
        or "text-embedding-3-small"
    )

    vector_index = None
    if not args.skip_vector:
        try:
            vector_index = create_vector_index(
                config.vector_backend, config.database_url, embedding_model
            )
        except Exception as exc:
            LOGGER.warning(
                "Failed to initialize vector index (%s); skipping vector cleanup", exc
            )

    removed: list[str] = []

    for i in range(args.count):
        candidates = list_non_leaf_roots(doc_store)
        if not candidates:
            LOGGER.info("No removable non-leaf roots remain after %d iteration(s)", i)
            break

        root = random.choice(candidates)
        LOGGER.info(
            "Selected root %s (height=%s, span=[%s,%s]) for removal",
            root.id,
            getattr(root, "height", "?"),
            root.span_start,
            root.span_end,
        )
        if args.dry_run:
            continue

        removed_id = remove_single_root(doc_store, vector_index, root)
        if not removed_id:
            LOGGER.info("No additional roots removed on iteration %d", i)
            break
        removed.append(removed_id)

    if not args.dry_run and removed:
        current_version = doc_store.get_version() or 1
        try:
            doc_store.set_metadata(version=current_version + 1)
            LOGGER.info("Bumped document version to %d", current_version + 1)
        except Exception as exc:
            LOGGER.warning("Failed to bump document version: %s", exc)

    if args.dry_run:
        LOGGER.info("Dry run complete. %d root(s) would be removed.", args.count)
    else:
        LOGGER.info(
            "Removed %d root(s): %s",
            len(removed),
            ", ".join(removed) if removed else "<none>",
        )


if __name__ == "__main__":
    main()
