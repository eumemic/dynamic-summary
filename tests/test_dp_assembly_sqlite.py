"""Example test using the real in-memory SQLite backend instead of mocks.

This mirrors a subset of tests from test_dp_assembly.py to demonstrate how to
use the new sqlite fixtures. Use this as a template for migrating additional
tests using SQLite backend.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from ragzoom.assemble import Assembler
from ragzoom.document_store import DocumentStore


@pytest.mark.usefixtures("sqlite_backend")
class TestDPAssemblySQLite:
    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> DocumentStore:
        # Create a document-scoped store for doc1
        return sqlite_store_factory("doc1")

    @pytest.fixture
    def assembler(self, doc_store: DocumentStore) -> Assembler:
        return Assembler(doc_store)

    @pytest.fixture
    def seed_nodes(self, doc_store: DocumentStore) -> None:
        """Create a small tree directly in the SQLite backend.

        Structure:
            root (0-82)
            /          \
         left(0-41)   right(41-82)
         /    \\        /     \
      leaf1 leaf2   leaf3  leaf4
        """
        nodes = [
            # Leaves
            {
                "node_id": "leaf1",
                "text": "First chunk of text.",
                "embedding": [],
                "span_start": 0,
                "span_end": 20,
                "document_id": "doc1",
                "token_count": 3,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf2",
                "text": "Second chunk of text.",
                "embedding": [],
                "span_start": 20,
                "span_end": 41,
                "document_id": "doc1",
                "token_count": 4,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf3",
                "text": "Third chunk of text.",
                "embedding": [],
                "span_start": 41,
                "span_end": 61,
                "document_id": "doc1",
                "token_count": 4,
                "height": 0,
                "level_index": 0,
            },
            {
                "node_id": "leaf4",
                "text": "Fourth chunk of text.",
                "embedding": [],
                "span_start": 61,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 4,
                "height": 0,
                "level_index": 0,
            },
            # Internal
            {
                "node_id": "left",
                "text": "Summary of first and second chunks.",
                "embedding": [],
                "span_start": 0,
                "span_end": 41,
                "document_id": "doc1",
                "token_count": 6,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf1",
                "right_child_id": "leaf2",
            },
            {
                "node_id": "right",
                "text": "Summary of third and fourth chunks.",
                "embedding": [],
                "span_start": 41,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 6,
                "height": 1,
                "level_index": 0,
                "left_child_id": "leaf3",
                "right_child_id": "leaf4",
            },
            {
                "node_id": "root",
                "text": "Overall document summary.",
                "embedding": [],
                "span_start": 0,
                "span_end": 82,
                "document_id": "doc1",
                "token_count": 4,
                "height": 2,
                "level_index": 0,
                "left_child_id": "left",
                "right_child_id": "right",
            },
        ]
        doc_store.nodes.add_batch(nodes)  # type: ignore[arg-type]
        # Update parent references
        doc_store.nodes.update_parent_references_batch(
            [
                ("leaf1", "left"),
                ("leaf2", "left"),
                ("leaf3", "right"),
                ("leaf4", "right"),
                ("left", "root"),
                ("right", "root"),
            ]
        )

    def test_basic_dp_assembly_sqlite(
        self, assembler: Assembler, seed_nodes: None
    ) -> None:
        tiling = ["leaf1", "leaf2"]
        result = assembler.assemble_dp(tiling)
        assert result == "First chunk of text.\n\nSecond chunk of text."

    def test_internal_node_assembly_sqlite(
        self, assembler: Assembler, seed_nodes: None
    ) -> None:
        tiling = ["left", "leaf3"]
        result = assembler.assemble_dp(tiling)
        assert result == "Summary of first and second chunks.\n\nThird chunk of text."
