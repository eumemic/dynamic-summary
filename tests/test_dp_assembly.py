"""Tests for DP assembly using backend-agnostic pattern.

This demonstrates how to use the storage backend pattern for DP assembly tests.
"""

from __future__ import annotations

import pytest

from ragzoom.assemble import Assembler
from ragzoom.contracts.storage_backend import StorageBackend


class TestDPAssembly:
    @pytest.fixture
    def seed_nodes(self, storage_backend: StorageBackend) -> None:
        """Create a small tree directly in the storage backend.

        Structure:
            root (0-82)
            /          \
         left(0-41)   right(41-82)
         /    \\        /     \
      leaf1 leaf2   leaf3  leaf4
        """
        # Get document store and set metadata
        doc_store = storage_backend.for_document("doc1")
        doc_store.set_metadata(
            file_path="dp_assembly_test.txt",
            content_hash="dp-assembly-test-hash",
            chunk_count=4,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        from typing import cast

        import numpy as np
        from numpy.typing import NDArray

        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = cast(
            list[
                dict[
                    str,
                    str | int | float | bool | list[float] | NDArray[np.float64] | None,
                ]
            ],
            [
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
                },
                # Internal
                {
                    "node_id": "left",
                    "text": "Summary of first and second chunks.",
                    "embedding": [],
                    "span_start": 0,
                    "span_end": 41,
                    "document_id": "doc1",
                    "height": 1,
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
                    "height": 1,
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
                    "height": 2,
                    "left_child_id": "left",
                    "right_child_id": "right",
                },
            ],
        )
        doc_store.nodes.add_batch(nodes)
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

    def test_basic_dp_assembly(
        self, storage_backend: StorageBackend, seed_nodes: None
    ) -> None:
        doc_store = storage_backend.for_document("doc1")
        assembler = Assembler(doc_store)
        tiling = ["leaf1", "leaf2"]
        result = assembler.assemble_dp(tiling)
        assert result == "First chunk of text.\n\nSecond chunk of text."

    def test_internal_node_assembly(
        self, storage_backend: StorageBackend, seed_nodes: None
    ) -> None:
        doc_store = storage_backend.for_document("doc1")
        assembler = Assembler(doc_store)
        tiling = ["left", "leaf3"]
        result = assembler.assemble_dp(tiling)
        assert result == "Summary of first and second chunks.\n\nThird chunk of text."
