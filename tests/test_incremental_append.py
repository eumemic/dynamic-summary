import asyncio
import os
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from ragzoom.config import IndexConfig
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.tree_node import TreeNode
from ragzoom.document_store import DocumentStore
from ragzoom.index import TreeBuilder
from ragzoom.splitter import TextSplitter
from ragzoom.validate import set_validation_enabled
from ragzoom.vector_factory import create_vector_index
from tests.conftest import BackwardCompatibilityConfig


def _make_tree_builder(
    doc_id: str,
    storage_backend: StorageBackend,
    index_config: IndexConfig,
    vector_backend: str,
    database_url: str,
    mock_client: MagicMock,
) -> tuple[TreeBuilder, Callable[[], list[tuple[int, int, int, str]]]]:
    try:
        storage_backend.clear_document(doc_id)
    except Exception:
        pass

    doc_store = storage_backend.for_document(doc_id)
    doc_store.set_metadata(
        file_path=f"{doc_id}.txt",
        content_hash="",
        chunk_count=0,
        embedding_model=index_config.embedding_model,
        summary_model=index_config.summary_model,
        version=1,
    )

    vector_index = create_vector_index(
        vector_backend,
        database_url,
        index_config.embedding_model,
    )
    builder = TreeBuilder(
        index_config,
        doc_store,
        vector_index=vector_index,
        max_concurrent=5,
    )
    builder.llm_service.client = mock_client

    def snapshot() -> list[tuple[int, int, int, str]]:
        nodes = doc_store.nodes.get_all()
        return sorted(
            [
                (
                    int(node.height),
                    int(node.span_start),
                    int(node.span_end),
                    node.text or "",
                )
                for node in nodes
            ]
        )

    return builder, snapshot


def _reconstruct_document(doc_store: DocumentStore) -> str:
    leaves = doc_store.nodes.get_leaves()
    leaves.sort(key=lambda n: int(n.span_start))
    return "".join(leaf.text or "" for leaf in leaves)


@pytest.fixture
def enable_incremental(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RAGZOOM_ENABLE_INCREMENTAL", "1")


class TestIncrementalAppend:
    @pytest.mark.slow_threshold(20.0)
    def test_incremental_equivalence(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        enable_incremental: None,
    ) -> None:
        config = base_config.index_config
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        full_builder, full_snapshot = _make_tree_builder(
            "doc-full",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        incremental_builder, incremental_snapshot = _make_tree_builder(
            "doc-incremental",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        full_text = "".join(
            [
                f"Paragraph {i}. This is deterministic content for testing.\n"
                for i in range(6)
            ]
        )

        asyncio.run(full_builder.add_document_async(full_text, show_progress=False))

        # Append the document in three segments (two paragraphs each)
        splitter = TextSplitter(config)
        chunks = splitter.split_text(full_text)
        first_chunk = chunks[0]
        remaining_text = full_text[len(first_chunk) :]
        midpoint = len(remaining_text) // 2
        pieces = [
            first_chunk,
            remaining_text[:midpoint],
            remaining_text[midpoint:],
        ]
        asyncio.run(
            incremental_builder.add_document_async(pieces[0], show_progress=False)
        )
        for part in pieces[1:]:
            asyncio.run(
                incremental_builder.append_text_async(
                    part,
                    show_progress=False,
                )
            )

        assert full_snapshot() == incremental_snapshot()

        full_doc = _reconstruct_document(full_builder.document_store)
        incremental_doc = _reconstruct_document(incremental_builder.document_store)
        assert incremental_doc == full_doc == full_text

        assert incremental_builder.document_store.get_version() == 1 + (len(pieces) - 1)

    def test_append_promotes_new_root(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        enable_incremental: None,
    ) -> None:
        config = base_config.index_config
        splitter = TextSplitter(config)
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        incremental_builder, incremental_snapshot = _make_tree_builder(
            "doc-root-incremental",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )
        full_builder, full_snapshot = _make_tree_builder(
            "doc-root-full",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        paragraphs = [
            f"Section {i}. Deterministic paragraph for root promotion testing.\n"
            for i in range(8)
        ]
        full_text = "".join(paragraphs)
        full_chunks = splitter.split_text(full_text)
        while len(full_chunks) < 4:
            start_index = len(paragraphs)
            paragraphs.extend(
                f"Section {start_index + j}. Additional deterministic text to force splits.\n"
                for j in range(4)
            )
            full_text = "".join(paragraphs)
            full_chunks = splitter.split_text(full_text)

        midpoint = len(full_chunks) // 2
        initial_text = "".join(full_chunks[:midpoint])
        append_text = "".join(full_chunks[midpoint:])

        asyncio.run(
            incremental_builder.add_document_async(initial_text, show_progress=False)
        )
        root_before = incremental_builder.document_store.tree.get_root()
        assert root_before is not None

        asyncio.run(
            incremental_builder.append_text_async(append_text, show_progress=False)
        )
        root_after = incremental_builder.document_store.tree.get_root()
        assert root_after is not None
        assert root_after.height >= root_before.height

        asyncio.run(full_builder.add_document_async(full_text, show_progress=False))

        assert incremental_snapshot() == full_snapshot()
        assert _reconstruct_document(incremental_builder.document_store) == full_text
        assert incremental_builder.document_store.get_version() == 2

    def test_small_append_fast_path_preserves_leaf(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        enable_incremental: None,
    ) -> None:
        config = base_config.index_config
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        builder, _ = _make_tree_builder(
            "doc-fast",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        initial_text = "Hello world."
        asyncio.run(builder.add_document_async(initial_text, show_progress=False))

        leaves_before = builder.document_store.nodes.get_leaves()
        assert len(leaves_before) == 1
        leaf_id = leaves_before[0].id

        asyncio.run(builder.append_text_async(" General Kenobi!", show_progress=False))

        leaves_after = builder.document_store.nodes.get_leaves()
        assert len(leaves_after) == 1
        assert leaves_after[0].id == leaf_id
        assert builder.document_store.get_version() == 2

        reconstructed = _reconstruct_document(builder.document_store)
        assert reconstructed == initial_text + " General Kenobi!"

    def test_unicode_append_preserves_text(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        enable_incremental: None,
    ) -> None:
        config = base_config.index_config
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        builder, _ = _make_tree_builder(
            "doc-unicode",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        base_text = "こんにちは世界。"
        asyncio.run(builder.add_document_async(base_text, show_progress=False))

        extra = "🌟追加のテキスト🌟"
        asyncio.run(builder.append_text_async(extra, show_progress=False))

        reconstructed = _reconstruct_document(builder.document_store)
        assert reconstructed == base_text + extra

    def test_append_telemetry_contains_metadata(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        enable_incremental: None,
    ) -> None:
        from unittest.mock import MagicMock

        config = base_config.index_config
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        builder, _ = _make_tree_builder(
            "doc-telemetry",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        base_text = "Telemetry baseline."
        asyncio.run(builder.add_document_async(base_text, show_progress=False))

        leaves_before = builder.document_store.nodes.get_leaves()
        leaves_before.sort(key=lambda n: int(n.span_start))
        right_leaf = leaves_before[-1]

        extra = " Additional telemetry slice."
        reporter = MagicMock()
        reporter.finalize.return_value = {"nodes": []}

        result = asyncio.run(
            builder.append_text_async(
                extra,
                show_progress=False,
                reporter=reporter,
            )
        )

        assert isinstance(result, tuple)
        _doc_id, telemetry = result
        assert telemetry.get("nodes") == []

        reporter.record_append_metadata.assert_called_once()
        metadata_kwargs = reporter.record_append_metadata.call_args.kwargs
        assert (
            metadata_kwargs["document_version"] == builder.document_store.get_version()
        )

        expected_start = int(right_leaf.span_start)
        expected_end = expected_start + len((right_leaf.text or "") + extra)
        assert metadata_kwargs["span_start"] == expected_start
        assert metadata_kwargs["span_end"] == expected_end
        assert metadata_kwargs["mutated_nodes"] >= metadata_kwargs["summary_nodes"] >= 0

    def test_append_rollback_on_vector_failure(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        enable_incremental: None,
    ) -> None:
        config = base_config.index_config
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        builder, _ = _make_tree_builder(
            "doc-fail-vector",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        initial = "Base document."
        asyncio.run(builder.add_document_async(initial, show_progress=False))

        def failing_upsert(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("boom")

        setattr(builder.vector_index, "upsert", failing_upsert)

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(builder.append_text_async(" will fail", show_progress=False))

        # Ensure version unchanged and content intact
        assert builder.document_store.get_version() == 1
        assert _reconstruct_document(builder.document_store) == initial

    def test_append_rollback_on_sql_failure(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
        enable_incremental: None,
    ) -> None:
        config = base_config.index_config
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        builder, _ = _make_tree_builder(
            "doc-fail-sql",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        initial = "Stable text."
        asyncio.run(builder.add_document_async(initial, show_progress=False))

        original_upsert = builder.document_store.nodes.upsert_nodes_batch

        def failing_upsert(
            payload: list[dict[str, object]],
            session: object | None = None,
        ) -> list[TreeNode]:
            raise RuntimeError("sql-fail")

        setattr(builder.document_store.nodes, "upsert_nodes_batch", failing_upsert)

        try:
            with pytest.raises(RuntimeError, match="sql-fail"):
                asyncio.run(
                    builder.append_text_async(" more text", show_progress=False)
                )
        finally:
            setattr(
                builder.document_store.nodes,
                "upsert_nodes_batch",
                original_upsert,
            )

        assert builder.document_store.get_version() == 1
        assert _reconstruct_document(builder.document_store) == initial


@pytest.mark.usefixtures("enable_incremental")
class TestAppendValidation:
    def test_validation_passes_on_correct_append(
        self,
        base_config: BackwardCompatibilityConfig,
        storage_backend: StorageBackend,
        mock_openai_async_client: MagicMock,
    ) -> None:
        config = base_config.index_config
        vector_backend = os.environ.get("RAGZOOM_VECTOR_BACKEND", "python")
        database_url = os.environ.get("RAGZOOM_DATABASE_URL", "sqlite:///:memory:")

        builder, _ = _make_tree_builder(
            "doc-validate",
            storage_backend,
            config,
            vector_backend,
            database_url,
            mock_openai_async_client,
        )

        asyncio.run(builder.add_document_async("seed text", show_progress=False))

        set_validation_enabled(True)
        try:
            asyncio.run(builder.append_text_async(" appended", show_progress=False))
        finally:
            set_validation_enabled(False)

        assert builder.document_store.get_version() == 2
