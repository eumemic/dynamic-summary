"""SQLite-backed tree structure validation tests using the runtime harness."""

from __future__ import annotations

from collections.abc import Callable, Generator
from typing import TypedDict, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.document_store import DocumentStore
from ragzoom.splitter import TextSplitter
from ragzoom.validate import set_validation_enabled, validate_tree_is_left_balanced
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
    left_child_id: str | None
    right_child_id: str | None


def _add_nodes(store: DocumentStore, nodes: list[NodePayload]) -> None:
    store.nodes.add_batch(
        cast(
            list[
                dict[
                    str,
                    str | int | float | bool | list[float] | NDArray[np.float64] | None,
                ]
            ],
            nodes,
        )
    )


def _configure_runtime(harness: IndexerRuntimeHarness, config: IndexConfig) -> None:
    harness.runtime._index_config = config
    harness.runtime._append_executor._config = config
    harness.runtime._append_executor._splitter = TextSplitter(config)
    harness.worker_coordinator._index_config = config
    harness.llm_service.config = config
    harness.telemetry_manager._index_config = config


@pytest.mark.usefixtures("sqlite_backend")
class TestTreeStructureSQLite:
    """Mirror of tree structure tests against the sqlite backend."""

    @pytest.fixture
    def doc_store(
        self, sqlite_store_factory: Callable[[str | None], DocumentStore]
    ) -> Generator[DocumentStore, None, None]:
        store = sqlite_store_factory("test-doc")
        store.set_metadata(
            file_path="sqlite-tree-validation.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        yield store
        sqlite_store_factory("test-doc")

    def test_manual_left_balanced_tree(self, doc_store: DocumentStore) -> None:
        nodes: list[NodePayload] = [
            NodePayload(
                node_id="L1",
                text="L1",
                embedding=[0.0],
                span_start=0,
                span_end=10,
                document_id="test-doc",
                height=0,
            ),
            NodePayload(
                node_id="L2",
                text="L2",
                embedding=[0.0],
                span_start=10,
                span_end=20,
                document_id="test-doc",
                height=0,
            ),
            NodePayload(
                node_id="P1",
                text="P1",
                embedding=[0.0],
                span_start=0,
                span_end=20,
                document_id="test-doc",
                height=1,
                left_child_id="L1",
                right_child_id="L2",
            ),
            NodePayload(
                node_id="root",
                text="root",
                embedding=[0.0],
                span_start=0,
                span_end=20,
                document_id="test-doc",
                height=2,
                left_child_id="P1",
                right_child_id=None,
            ),
        ]
        _add_nodes(doc_store, nodes)
        doc_store.nodes.update_parent_references_batch(
            [("L1", "P1"), ("L2", "P1"), ("P1", "root")]
        )
        assert validate_tree_is_left_balanced(doc_store) is None

    @pytest.mark.asyncio
    async def test_runtime_indexing_validates(
        self,
        base_config: BackwardCompatibilityConfig,
        sqlite_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        document_id = "sqlite-tree"
        config = base_config.index_config.replace(
            target_chunk_tokens=18, preceding_context_tokens=4
        )
        _configure_runtime(indexer_runtime_harness, config)

        sqlite_backend.clear_document(document_id)
        await indexer_runtime_harness.clear(document_id)

        await indexer_runtime_harness.append(
            document_id,
            " ".join([f"Sentence {i}." for i in range(10)]),
            replace_existing=True,
            file_path="sqlite-tree-validation.txt",
        )

        doc_store = indexer_runtime_harness.runtime._store.for_document(document_id)
        assert validate_tree_is_left_balanced(doc_store) is None

    @pytest.mark.asyncio
    async def test_runtime_validation_toggle(
        self,
        base_config: BackwardCompatibilityConfig,
        sqlite_backend: StorageBackend,
        indexer_runtime_harness: IndexerRuntimeHarness,
    ) -> None:
        document_id = "sqlite-tree-toggle"
        config = base_config.index_config
        _configure_runtime(indexer_runtime_harness, config)

        sqlite_backend.clear_document(document_id)
        await indexer_runtime_harness.clear(document_id)

        test_text = "Paragraph A.\n\nParagraph B.\n\nParagraph C."

        set_validation_enabled(True)
        try:
            await indexer_runtime_harness.append(
                document_id,
                test_text,
                replace_existing=True,
                file_path="sqlite-tree-toggle.txt",
            )
        finally:
            set_validation_enabled(False)

        doc_store = indexer_runtime_harness.runtime._store.for_document(document_id)
        assert validate_tree_is_left_balanced(doc_store) is None
