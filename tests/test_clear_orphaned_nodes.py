"""Backend-agnostic test for automatic clearing of orphaned nodes from interrupted indexing."""

import os
import tempfile
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ragzoom.cli import cli
from ragzoom.config import (
    IndexConfig,
    OperationalConfig,
    PrecedingContextConfig,
    PrecedingContextSettings,
    SecretStr,
)
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.services.indexing_service import IndexingResult


class TestAutomaticClearing:
    """Test that automatic clearing properly handles orphaned nodes from interrupted indexing."""

    @pytest.fixture
    def temp_db(self, tmp_path: object) -> Generator[str, None, None]:
        """Create a temporary database for testing."""
        from pathlib import Path

        tmp_path_obj = (
            Path(str(tmp_path)) if not isinstance(tmp_path, Path) else tmp_path
        )
        db_path = tmp_path_obj / "test_ragzoom.db"
        original_db = os.environ.get("RAGZOOM_DB_PATH")
        os.environ["RAGZOOM_DB_PATH"] = str(db_path)
        yield str(db_path)
        if original_db:
            os.environ["RAGZOOM_DB_PATH"] = original_db
        else:
            os.environ.pop("RAGZOOM_DB_PATH", None)

    @pytest.fixture
    def config(self) -> IndexConfig:
        """Create a test configuration."""
        return IndexConfig(
            target_chunk_tokens=200,
            target_embedding_context_tokens=200,
            max_parallelism=30,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            retry_threshold=0.2,
            max_retries=0,
            embedding_batch_size=100,
            use_anti_verbatim_vaccine=True,
            processing_strategy="bottom_to_top",
            preceding_context=PrecedingContextSettings(
                leaf=PrecedingContextConfig(
                    verbatim_tokens=0, min_forest_completeness=0.5
                ),
                inner=PrecedingContextConfig(
                    verbatim_tokens=0,
                    min_forest_completeness=0.5,
                    token_cap=0,  # Cap inner node context
                ),
            ),
        )

    @pytest.fixture
    def operational_config(self, temp_db: str) -> OperationalConfig:
        """Create operational configuration for test database."""
        return OperationalConfig(
            openai_api_key=SecretStr("test-key"),
            database_url=f"postgresql:///{temp_db}",
            cache_size=100,
        )

    def simulate_interrupted_indexing(
        self, storage_backend: StorageBackend, document_id: str, num_nodes: int = 248
    ) -> None:
        """Simulate an interrupted indexing run that leaves orphaned nodes.

        This simulates what happens when indexing is interrupted after storing nodes
        but before creating the Document record.
        """
        doc_store = storage_backend.for_document(document_id)

        # Create orphaned nodes (as would happen during interrupted indexing)
        nodes_data = []
        for i in range(num_nodes):
            span_start = i * 100
            span_end = (i + 1) * 100
            nodes_data.append(
                {
                    "node_id": f"node_{i}",
                    "text": f"Text content {i}",  # Dummy embedding
                    "span_start": span_start,
                    "span_end": span_end,
                    "document_id": document_id,
                    "token_count": 50,
                    "height": 0,
                    "level_index": i,
                }
            )

        from typing import cast

        # Convert to properly typed format
        typed_nodes_data: list[NodeDataDict] = cast(
            list[NodeDataDict],
            nodes_data,
        )
        doc_store.nodes.add_batch(typed_nodes_data)

        # Important: Do NOT create a Document record via set_metadata
        # This simulates interruption before the Document record is created
        # (which happens at the end of indexing)

    def test_automatic_clearing_deletes_orphaned_nodes(
        self,
        temp_db: str,
        config: IndexConfig,
        operational_config: OperationalConfig,
        storage_backend: StorageBackend,
    ) -> None:
        """Indexing should remove orphaned nodes left by an interrupted run."""
        runner = CliRunner()
        document_id = "test_document.txt"

        self.simulate_interrupted_indexing(storage_backend, document_id, num_nodes=248)
        orphaned_count = len(storage_backend.for_document(document_id).nodes.get_all())
        assert orphaned_count == 248

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for indexing.")
            temp_file = f.name

        try:
            with patch("ragzoom.cli.GrpcRagzoomClient") as mock_client_cls:
                client = MagicMock()
                client.__enter__.return_value = client
                client.__exit__.return_value = None

                def append_text_side_effect(
                    *,
                    document_id: str,
                    content: bytes,
                    collect_telemetry: bool,
                ) -> IndexingResult:
                    storage_backend.clear_document(document_id)
                    doc_store = storage_backend.for_document(document_id)
                    doc_store.set_metadata(
                        file_path=document_id,
                        embedding_model="text-embedding-3-small",
                        summary_model="gpt-4o-mini",
                    )
                    doc_store.nodes.add_batch(
                        [
                            {
                                "node_id": "mock_root",
                                "text": "Mock root node",
                                "span_start": 0,
                                "span_end": 100,
                                "document_id": document_id,
                                "token_count": 50,
                                "height": 0,
                                "level_index": 0,
                            }
                        ]
                    )
                    return IndexingResult(
                        document_id=document_id,
                        chunks_created=1,
                        tree_depth=1,
                        mutated_nodes=1,
                        resummarized_nodes=0,
                        new_leaves=1,
                        telemetry=None,
                    )

                client.append_text.side_effect = append_text_side_effect
                mock_client_cls.return_value = client

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    result = runner.invoke(
                        cli, ["index", temp_file, "--document-id", document_id]
                    )
                    assert result.exit_code == 0, result.output

            final_store = storage_backend.for_document(document_id)
            node_ids = {node.id for node in final_store.nodes.get_all()}
            assert not any(node_id.startswith("node_") for node_id in node_ids)
            assert node_ids == {"mock_root"}
        finally:
            os.unlink(temp_file)

    def test_automatic_clearing_works_with_document_record(
        self,
        temp_db: str,
        config: IndexConfig,
        operational_config: OperationalConfig,
        storage_backend: StorageBackend,
    ) -> None:
        """Indexing should replace prior data even when metadata already exists."""
        runner = CliRunner()
        document_id = "test_document.txt"

        self.simulate_interrupted_indexing(storage_backend, document_id, num_nodes=248)
        doc_store = storage_backend.for_document(document_id)
        doc_store.set_metadata(
            file_path="/test/path.txt",
            embedding_model="test-model",
            summary_model="test-model",
        )
        assert len(doc_store.nodes.get_all()) == 248

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for indexing.")
            temp_file = f.name

        try:
            with patch("ragzoom.cli.GrpcRagzoomClient") as mock_client_cls:
                client = MagicMock()
                client.__enter__.return_value = client
                client.__exit__.return_value = None

                def append_text_side_effect(
                    *,
                    document_id: str,
                    content: bytes,
                    collect_telemetry: bool,
                ) -> IndexingResult:
                    storage_backend.clear_document(document_id)
                    refreshed_store = storage_backend.for_document(document_id)
                    refreshed_store.set_metadata(
                        file_path=document_id,
                        embedding_model="text-embedding-3-small",
                        summary_model="gpt-4o-mini",
                    )
                    refreshed_store.nodes.add_batch(
                        [
                            {
                                "node_id": "mock_root",
                                "text": "Mock root node",
                                "span_start": 0,
                                "span_end": 100,
                                "document_id": document_id,
                                "token_count": 50,
                                "height": 0,
                                "level_index": 0,
                            }
                        ]
                    )
                    return IndexingResult(
                        document_id=document_id,
                        chunks_created=1,
                        tree_depth=1,
                        mutated_nodes=1,
                        resummarized_nodes=0,
                        new_leaves=1,
                        telemetry=None,
                    )

                client.append_text.side_effect = append_text_side_effect
                mock_client_cls.return_value = client

                with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                    result = runner.invoke(
                        cli, ["index", temp_file, "--document-id", document_id]
                    )
                    assert result.exit_code == 0, result.output

            refreshed_store = storage_backend.for_document(document_id)
            node_ids = {node.id for node in refreshed_store.nodes.get_all()}
            assert node_ids == {"mock_root"}
        finally:
            os.unlink(temp_file)
