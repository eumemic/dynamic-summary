"""Tree structure validation tests using the runtime harness."""

from __future__ import annotations

from collections.abc import Generator
from typing import TypedDict, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import IndexConfig
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from ragzoom.validate import set_validation_enabled, validate_perfect_binary_trees
from tests.conftest import BackwardCompatibilityConfig, IndexerRuntimeHarness


class NodePayload(TypedDict, total=False):
    node_id: str
    text: str
    embedding: list[float] | NDArray[np.float64]
    span_start: int
    span_end: int
    document_id: str
    height: int
    token_count: int
    level_index: int
    left_child_id: str | None
    right_child_id: str | None


@pytest.fixture
def doc_store(storage_backend: StorageBackend) -> Generator[DocumentStore, None, None]:
    document_id = "test-doc"
    storage_backend.clear_document(document_id)
    store = storage_backend.for_document(document_id)
    store.set_metadata(
        file_path="tree-validation.txt",
        embedding_model="text-embedding-3-small",
        summary_model="gpt-4o-mini",
    )
    yield store
    storage_backend.clear_document(document_id)


def _add_nodes(store: DocumentStore, nodes: list[NodePayload]) -> None:
    store.nodes.add_batch(
        cast(
            list[NodeDataDict],
            nodes,
        )
    )


def _configure_runtime(harness: IndexerRuntimeHarness, config: IndexConfig) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.indexing_engine._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config


class TestTreeValidation:
    """Unit tests for perfect binary tree validation on manually constructed trees."""

    def test_full_tree_passes_validation(self, doc_store: DocumentStore) -> None:
        nodes: list[NodePayload] = [
            NodePayload(
                node_id="L1",
                text="L1",
                embedding=[0.0],
                span_start=0,
                span_end=10,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=0,
            ),
            NodePayload(
                node_id="L2",
                text="L2",
                embedding=[0.0],
                span_start=10,
                span_end=20,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=1,
            ),
            NodePayload(
                node_id="L3",
                text="L3",
                embedding=[0.0],
                span_start=20,
                span_end=30,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=2,
            ),
            NodePayload(
                node_id="L4",
                text="L4",
                embedding=[0.0],
                span_start=30,
                span_end=40,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=3,
            ),
            NodePayload(
                node_id="P1",
                text="P1",
                embedding=[0.0],
                span_start=0,
                span_end=20,
                document_id="test-doc",
                height=1,
                token_count=4,
                level_index=0,
                left_child_id="L1",
                right_child_id="L2",
            ),
            NodePayload(
                node_id="P2",
                text="P2",
                embedding=[0.0],
                span_start=20,
                span_end=40,
                document_id="test-doc",
                height=1,
                token_count=4,
                level_index=1,
                left_child_id="L3",
                right_child_id="L4",
            ),
            NodePayload(
                node_id="root",
                text="root",
                embedding=[0.0],
                span_start=0,
                span_end=40,
                document_id="test-doc",
                height=2,
                token_count=8,
                level_index=0,
                left_child_id="P1",
                right_child_id="P2",
            ),
        ]
        _add_nodes(doc_store, nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("L1", "P1"),
                ("L2", "P1"),
                ("L3", "P2"),
                ("L4", "P2"),
                ("P1", "root"),
                ("P2", "root"),
            ]
        )
        assert validate_perfect_binary_trees(doc_store) is None

    def test_left_only_child_fails_validation(self, doc_store: DocumentStore) -> None:
        """A node with only a left child violates perfect binary tree invariant."""
        nodes: list[NodePayload] = [
            NodePayload(
                node_id="L1",
                text="L1",
                embedding=[0.0],
                span_start=0,
                span_end=10,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=0,
            ),
            NodePayload(
                node_id="L2",
                text="L2",
                embedding=[0.0],
                span_start=10,
                span_end=20,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=1,
            ),
            NodePayload(
                node_id="L3",
                text="L3",
                embedding=[0.0],
                span_start=20,
                span_end=30,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=2,
            ),
            NodePayload(
                node_id="P1",
                text="P1",
                embedding=[0.0],
                span_start=0,
                span_end=20,
                document_id="test-doc",
                height=1,
                token_count=4,
                level_index=0,
                left_child_id="L1",
                right_child_id="L2",
            ),
            NodePayload(
                node_id="P2",
                text="P2",
                embedding=[0.0],
                span_start=20,
                span_end=30,
                document_id="test-doc",
                height=1,
                token_count=2,
                level_index=1,
                left_child_id="L3",
                right_child_id=None,  # Only left child - violates perfect binary tree
            ),
            NodePayload(
                node_id="root",
                text="root",
                embedding=[0.0],
                span_start=0,
                span_end=30,
                document_id="test-doc",
                height=2,
                token_count=6,
                level_index=0,
                left_child_id="P1",
                right_child_id="P2",
            ),
        ]
        _add_nodes(doc_store, nodes)
        doc_store.nodes.update_parent_references_batch(
            [
                ("L1", "P1"),
                ("L2", "P1"),
                ("L3", "P2"),
                ("P1", "root"),
                ("P2", "root"),
            ]
        )
        result = validate_perfect_binary_trees(doc_store)
        assert result is not None and "only a left child" in result

    def test_invalid_child_reference_fails(self, doc_store: DocumentStore) -> None:
        _add_nodes(
            doc_store,
            [
                NodePayload(
                    node_id="root",
                    text="root",
                    embedding=[0.0],
                    span_start=0,
                    span_end=20,
                    document_id="test-doc",
                    height=1,
                    token_count=4,
                    level_index=0,
                    left_child_id="missing-left",
                    right_child_id="missing-right",
                )
            ],
        )
        result = validate_perfect_binary_trees(doc_store)
        assert result is not None and "non-existent" in result

    def test_right_child_without_left_child_fails(
        self, doc_store: DocumentStore
    ) -> None:
        nodes: list[NodePayload] = [
            NodePayload(
                node_id="L1",
                text="L1",
                embedding=[0.0],
                span_start=0,
                span_end=10,
                document_id="test-doc",
                height=0,
                token_count=2,
                level_index=0,
            ),
            NodePayload(
                node_id="P1",
                text="P1",
                embedding=[0.0],
                span_start=10,
                span_end=20,
                document_id="test-doc",
                height=1,
                token_count=2,
                level_index=0,
                left_child_id="L1",
                right_child_id=None,
            ),
            NodePayload(
                node_id="root",
                text="root",
                embedding=[0.0],
                span_start=0,
                span_end=20,
                document_id="test-doc",
                height=2,
                token_count=4,
                level_index=0,
                left_child_id=None,
                right_child_id="P1",
            ),
        ]
        _add_nodes(doc_store, nodes)
        doc_store.nodes.update_parent_references_batch([("L1", "P1"), ("P1", "root")])
        result = validate_perfect_binary_trees(doc_store)
        # Tree has nodes with single children (root has only right, P1 has only left)
        # Validation catches one of them
        assert result is not None and "not a perfect binary tree" in result


@pytest.mark.usefixtures("sqlite_backend")
class TestIndexingCreatesValidTrees:
    """Ensure runtime indexing produces perfect binary trees."""

    @pytest.mark.asyncio
    async def test_indexing_produces_valid_tree(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        document_id = "tree-validation"
        config = base_config.index_config.replace(
            target_chunk_tokens=20,
            embedding_batch_size=2,
        )
        _configure_runtime(indexer_runtime_harness, config)

        storage_backend.clear_document(document_id)
        await indexer_runtime_harness.clear(document_id)

        test_text = " ".join([f"Sentence {i}." for i in range(12)])
        await indexer_runtime_harness.append(
            document_id,
            test_text,
            replace_existing=True,
            file_path="tree-validation.txt",
        )

        doc_store = indexer_runtime_harness.runtime._store.for_document(document_id)
        assert validate_perfect_binary_trees(doc_store) is None

    @pytest.mark.asyncio
    async def test_validation_toggle_during_indexing(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        document_id = "tree-validation-toggle"
        config = base_config.index_config.replace(
            target_chunk_tokens=25,
        )
        _configure_runtime(indexer_runtime_harness, config)

        storage_backend.clear_document(document_id)
        await indexer_runtime_harness.clear(document_id)

        test_text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

        set_validation_enabled(True)
        try:
            await indexer_runtime_harness.append(
                document_id,
                test_text,
                replace_existing=True,
                file_path="tree-validation-toggle.txt",
            )
        finally:
            set_validation_enabled(False)

        doc_store = indexer_runtime_harness.runtime._store.for_document(document_id)
        assert validate_perfect_binary_trees(doc_store) is None
