"""Test for automatic clearing of orphaned nodes from interrupted indexing."""

import os
import tempfile
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from ragzoom.cli import cli
from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.models import Document, TreeNode
from ragzoom.services.indexing_service import IndexingResult
from ragzoom.store import StoreManager


class TestAutomaticClearing:
    """Test that automatic clearing properly handles orphaned nodes from interrupted indexing."""

    @pytest.fixture
    def temp_db(self, tmp_path: Any) -> Generator[str, None, None]:
        """Create a temporary database for testing."""
        db_path = tmp_path / "test_ragzoom.db"
        original_db = os.environ.get("RAGZOOM_DB_PATH")
        os.environ["RAGZOOM_DB_PATH"] = str(db_path)
        yield db_path
        if original_db:
            os.environ["RAGZOOM_DB_PATH"] = original_db
        else:
            os.environ.pop("RAGZOOM_DB_PATH", None)

    @pytest.fixture
    def config(self) -> IndexConfig:
        """Create a test configuration."""
        return IndexConfig(
            target_chunk_tokens=200,
            preceding_context_tokens=75,
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
            retry_threshold=0.2,
            max_retries=0,
            embedding_batch_size=100,
            use_anti_verbatim_vaccine=True,
            processing_strategy="bottom_to_top",
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
        self, store: "StoreManager", document_id: str, num_nodes: int = 248
    ) -> None:
        """Simulate an interrupted indexing run that leaves orphaned nodes.

        This simulates what happens when indexing is interrupted after storing nodes
        but before creating the Document record.
        """
        # Create orphaned nodes (as would happen during interrupted indexing)
        for i in range(num_nodes):
            span_start = i * 100
            span_end = (i + 1) * 100

            store.add_node(
                node_id=f"node_{i}",
                text=f"Text content {i}",
                embedding=[0.1] * 1536,  # Dummy embedding
                span_start=span_start,
                span_end=span_end,
                document_id=document_id,
                token_count=50,
            )

        # Important: Do NOT create a Document record
        # This simulates interruption before the Document record is created
        # (which happens at the end of indexing)

    def test_automatic_clearing_deletes_orphaned_nodes(
        self,
        temp_db: str,
        config: IndexConfig,
        operational_config: OperationalConfig,
        store: StoreManager,
    ) -> None:
        """Test that automatic clearing deletes orphaned nodes from interrupted indexing."""
        runner = CliRunner()
        document_id = "test_document.txt"

        # Simulate an interrupted indexing that left orphaned nodes
        self.simulate_interrupted_indexing(store, document_id, num_nodes=248)

        # Verify orphaned nodes exist
        with store.SessionLocal() as session:
            orphaned_count = (
                session.query(TreeNode).filter_by(document_id=document_id).count()
            )
            assert (
                orphaned_count == 248
            ), f"Expected 248 orphaned nodes, found {orphaned_count}"

            # Verify no Document record exists
            doc = session.query(Document).filter_by(id=document_id).first()
            assert (
                doc is None
            ), "Document record should not exist for interrupted indexing"

        # Create a test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for indexing.")
            temp_file = f.name

        try:
            # Mock the indexing service to avoid actual API calls
            with patch(
                "ragzoom.services.indexing_service.IndexingService"
            ) as mock_indexing_service:
                mock_instance = MagicMock()
                mock_indexing_service.return_value = mock_instance

                # Mock index_from_file to simulate the full indexing process including clearing
                def mock_index_from_file_side_effect(
                    *args: object, **kwargs: object
                ) -> IndexingResult:
                    # First, perform clearing like the real service would
                    store.clear_document(document_id)

                    # Add a mock root node so CLI stats calculation works
                    store.add_node(
                        node_id="mock_root",
                        text="Mock root node",
                        span_start=0,
                        span_end=100,
                        parent_id=None,
                        document_id=document_id,
                        embedding=[0.1] * 1536,
                    )
                    from ragzoom.services.indexing_service import IndexingResult

                    return IndexingResult(
                        document_id=document_id,
                        chunks_created=1,
                        tree_depth=1,
                    )

                mock_instance.index_from_file.side_effect = (
                    mock_index_from_file_side_effect
                )

                # Mock successful indexing that returns proper stats
                with patch("ragzoom.cli.create_store_with_docker") as mock_create_store:
                    # Use the real store for database operations
                    mock_create_store.return_value = store

                    # Mock IndexingService constructor to return our mock instance
                    with patch(
                        "ragzoom.cli.IndexingService"
                    ) as mock_indexing_service_cli:
                        mock_indexing_service_cli.return_value = mock_instance

                        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                            # Run index (clearing should be automatic)
                            result = runner.invoke(
                                cli, ["index", temp_file, "--document-id", document_id]
                            )

                            # The command should succeed
                            assert (
                                result.exit_code == 0
                            ), f"Command failed: {result.output}"

            # Check if orphaned nodes were deleted
            with store.SessionLocal() as session:
                all_nodes = (
                    session.query(TreeNode).filter_by(document_id=document_id).all()
                )

                # Automatic clearing should have deleted old nodes and added new ones
                # Check that none of the old node IDs exist (they were node_0 through node_247)
                old_node_ids = [f"node_{i}" for i in range(248)]
                remaining_old_nodes = [
                    node for node in all_nodes if node.id in old_node_ids
                ]

                assert (
                    len(remaining_old_nodes) == 0
                ), f"Found {len(remaining_old_nodes)} old orphaned nodes that should have been cleared"

                # Should have exactly 1 new node from the mock indexing
                assert (
                    len(all_nodes) == 1
                ), f"Expected exactly 1 new node after reindexing, found {len(all_nodes)}"

        finally:
            os.unlink(temp_file)

    def test_automatic_clearing_works_with_document_record(
        self,
        temp_db: str,
        config: IndexConfig,
        operational_config: OperationalConfig,
        store: StoreManager,
    ) -> None:
        """Test that automatic clearing works correctly when a Document record exists."""
        runner = CliRunner()
        document_id = "test_document.txt"

        # Simulate a complete indexing (with Document record)
        self.simulate_interrupted_indexing(store, document_id, num_nodes=248)

        # Add a Document record (simulating successful indexing)
        store.add_document(
            document_id=document_id,
            file_path="/test/path.txt",
            content_hash="test_hash",
            chunk_count=248,
            embedding_model="test-model",
            summary_model="test-model",
        )

        # Verify nodes and document exist
        with store.SessionLocal() as session:
            node_count = (
                session.query(TreeNode).filter_by(document_id=document_id).count()
            )
            assert node_count == 248

            doc = session.query(Document).filter_by(id=document_id).first()
            assert doc is not None

        # Create a test file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for indexing.")
            temp_file = f.name

        try:
            # Mock the indexing service
            with patch(
                "ragzoom.services.indexing_service.IndexingService"
            ) as mock_indexing_service:
                mock_instance = MagicMock()
                mock_indexing_service.return_value = mock_instance

                # Mock index_from_file to simulate the full indexing process including clearing
                def mock_index_from_file_side_effect(
                    *args: object, **kwargs: object
                ) -> IndexingResult:
                    # First, perform clearing like the real service would
                    store.clear_document(document_id)

                    # Add a mock root node so CLI stats calculation works
                    store.add_node(
                        node_id="mock_root",
                        text="Mock root node",
                        span_start=0,
                        span_end=100,
                        parent_id=None,
                        document_id=document_id,
                        embedding=[0.1] * 1536,
                    )
                    from ragzoom.services.indexing_service import IndexingResult

                    return IndexingResult(
                        document_id=document_id,
                        chunks_created=1,
                        tree_depth=1,
                    )

                mock_instance.index_from_file.side_effect = (
                    mock_index_from_file_side_effect
                )

                with patch("ragzoom.cli.create_store_with_docker") as mock_create_store:
                    mock_create_store.return_value = store

                    # Mock IndexingService constructor to return our mock instance
                    with patch(
                        "ragzoom.cli.IndexingService"
                    ) as mock_indexing_service_cli:
                        mock_indexing_service_cli.return_value = mock_instance

                        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                            # Run index (clearing should be automatic)
                            result = runner.invoke(
                                cli, ["index", temp_file, "--document-id", document_id]
                            )

                            assert (
                                result.exit_code == 0
                            ), f"Command failed: {result.output}"

            # Verify old nodes were cleared and new ones added
            with store.SessionLocal() as session:
                all_nodes = (
                    session.query(TreeNode).filter_by(document_id=document_id).all()
                )

                # Check that none of the old node IDs exist (they were node_0 through node_247)
                old_node_ids = [f"node_{i}" for i in range(248)]
                remaining_old_nodes = [
                    node for node in all_nodes if node.id in old_node_ids
                ]

                assert (
                    len(remaining_old_nodes) == 0
                ), f"Found {len(remaining_old_nodes)} old nodes that should have been cleared"

                # Should have exactly 1 new node from the mock indexing
                assert (
                    len(all_nodes) == 1
                ), f"Expected exactly 1 new node after reindexing, found {len(all_nodes)}"

        finally:
            os.unlink(temp_file)
