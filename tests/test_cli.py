"""Tests for CLI functionality."""

import os
import tempfile
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from ragzoom.cli import cli


class TestCLI:
    """Test the CLI commands."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_ragzoom(self):
        """Mock RagZoom components."""
        with (
            patch("ragzoom.cli.RagZoomConfig") as mock_config,
            patch("ragzoom.cli.Store") as mock_store,
            patch("ragzoom.cli.TreeBuilder") as mock_builder,
            patch("ragzoom.cli.Retriever") as mock_retriever,
            patch("ragzoom.cli.Assembler") as mock_assembler,
        ):

            # Mock config
            config_instance = Mock()
            config_instance.budget_tokens = 8000
            config_instance.mmr_lambda = 0.7
            config_instance.leaf_tokens = 200
            config_instance.slope_cap = True
            config_instance.smoothing_pass_enabled = False
            mock_config.return_value = config_instance

            # Mock store
            store_instance = Mock()
            store_instance.get_leaf_nodes.return_value = [
                Mock(id=f"node-{i}") for i in range(5)
            ]
            store_instance.get_root_node.return_value = Mock(depth=3)
            store_instance.get_pinned_nodes.return_value = []
            store_instance.collection.count.return_value = 10

            # Mock SessionLocal for database queries
            mock_session = Mock()
            mock_query = Mock()

            # Mock leaf nodes query
            leaf_nodes = [
                Mock(
                    id=f"leaf-{i}",
                    span_start=i * 100,
                    span_end=(i + 1) * 100,
                    text=f"text-{i}",
                    summary=None,
                )
                for i in range(5)
            ]
            mock_query.filter_by.return_value.all.return_value = leaf_nodes

            # Mock root node query
            root_node = Mock(id="root", depth=3, parent_id=None)
            mock_query.filter_by.return_value.first.return_value = root_node

            mock_session.query.return_value = mock_query
            mock_context_manager = Mock()
            mock_context_manager.__enter__ = Mock(return_value=mock_session)
            mock_context_manager.__exit__ = Mock(return_value=None)
            store_instance.SessionLocal.return_value = mock_context_manager

            mock_store.return_value = store_instance

            # Mock tree builder
            builder_instance = Mock()
            # add_document is called synchronously in CLI
            builder_instance.add_document.return_value = "doc-123"
            mock_builder.return_value = builder_instance

            # Mock retriever
            retriever_instance = Mock()
            retriever_instance.retrieve.return_value = Mock(
                node_ids=["node-1", "node-2"],
                frontier_nodes=["node-1", "node-2"],
                coverage_map={"node-1": 1.0, "node-2": 1.0},
            )
            mock_retriever.return_value = retriever_instance

            # Mock assembler
            assembler_instance = Mock()
            assembler_instance.assemble.return_value = (
                "This is a summary of the content."
            )
            assembler_instance.assemble_with_budget.return_value = (
                "This is a summary of the content.",
                100,
            )
            mock_assembler.return_value = assembler_instance

            yield {
                "config": mock_config,
                "store": mock_store,
                "builder": mock_builder,
                "retriever": mock_retriever,
                "assembler": mock_assembler,
                "config_instance": config_instance,
                "store_instance": store_instance,
                "builder_instance": builder_instance,
                "retriever_instance": retriever_instance,
                "assembler_instance": assembler_instance,
            }

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "RagZoom: Incremental, hierarchical RAG memory system." in result.output
        assert "Commands:" in result.output

    def test_status_command(self, runner, mock_ragzoom):
        """Test status command."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "SYSTEM STATUS" in result.output
            assert "Total nodes: 10" in result.output
            assert "Leaf nodes: 5" in result.output
            assert "Tree depth: 3" in result.output

    def test_index_command_with_file(self, runner, mock_ragzoom):
        """Test indexing a file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for indexing.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file])

                assert result.exit_code == 0
                assert "Document indexed successfully!" in result.output
                assert "doc-123" in result.output

                # Verify add_document was called with the content
                mock_ragzoom["builder_instance"].add_document.assert_called_once()
                call_args = mock_ragzoom["builder_instance"].add_document.call_args
                assert "Test content for indexing." in call_args[0][0]
        finally:
            os.unlink(temp_file)

    def test_index_command_with_text(self, runner, mock_ragzoom):
        """Test indexing with document ID."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(
                    cli, ["index", temp_file, "--document-id", "my-doc-id"]
                )

                assert result.exit_code == 0
                assert "Document indexed successfully!" in result.output
                assert "doc-123" in result.output

                # Verify add_document was called with document_id
                call_args = mock_ragzoom["builder_instance"].add_document.call_args
                assert call_args[1]["document_id"] == "my-doc-id"
        finally:
            os.unlink(temp_file)

    def test_query_command(self, runner, mock_ragzoom):
        """Test query command."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli, ["query", "Tell me about cats", "-d", "test-doc"]
            )

            assert result.exit_code == 0
            assert "SUMMARY:" in result.output
            assert "This is a summary of the content." in result.output

    def test_query_with_options(self, runner, mock_ragzoom):
        """Test query command with options."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli,
                [
                    "query",
                    "Tell me about cats",
                    "-d",
                    "test-doc",
                    "--n-max",
                    "5",
                    "--token-budget",
                    "1000",
                ],
            )

            assert result.exit_code == 0

            # Verify retrieve was called with correct query, n_max, budget_tokens, and document_id
            mock_ragzoom["retriever_instance"].retrieve.assert_called_once_with(
                "Tell me about cats",
                n_max=5,
                budget_tokens=1000,
                document_id="test-doc",
            )

    def test_pin_command(self, runner, mock_ragzoom):
        """Test pin command."""
        mock_ragzoom["store_instance"].pin_node.return_value = True

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["pin", "node-123"])

            assert result.exit_code == 0
            assert "✅ Node node-123 pinned successfully!" in result.output

    def test_pin_command_failure(self, runner, mock_ragzoom):
        """Test pin command when pinning fails."""
        mock_ragzoom["store_instance"].pin_node.return_value = False

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["pin", "node-999"])

            # Pin command returns exit code 1 on failure
            assert result.exit_code == 1
            assert "Failed to pin node node-999" in result.output

    def test_serve_command(self, runner, mock_ragzoom):
        """Test serve command."""
        with patch("uvicorn.run") as mock_uvicorn:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["serve", "--port", "8080"])

                assert result.exit_code == 0

                # Verify uvicorn was called with correct parameters
                mock_uvicorn.assert_called_once_with(
                    "ragzoom.api:app", host="0.0.0.0", port=8080, reload=False
                )

    def test_documents_command(self, runner, mock_ragzoom):
        """Test documents command."""
        # Mock document results
        from datetime import datetime
        from unittest.mock import MagicMock

        mock_doc1 = MagicMock()
        mock_doc1.id = "doc-123"
        mock_doc1.file_path = "/path/to/file.txt"
        mock_doc1.indexed_at = datetime.now()
        mock_doc1.chunk_count = 10

        mock_session = MagicMock()
        mock_session.query.return_value.all.return_value = [mock_doc1]
        mock_session.query.return_value.filter_by.return_value.count.return_value = 15

        mock_ragzoom[
            "store_instance"
        ].SessionLocal.return_value.__enter__.return_value = mock_session

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["documents"])

            assert result.exit_code == 0
            assert "Document ID: doc-123" in result.output
            assert "File: /path/to/file.txt" in result.output
            assert "Chunks: 10" in result.output

    def test_missing_api_key(self, runner):
        """Test commands fail without API key."""
        # Mock the RagZoomConfig to raise an error when instantiated without API key
        with patch("ragzoom.cli.RagZoomConfig") as mock_config:
            mock_config.side_effect = ValueError("Field required: openai_api_key")

            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
                result = runner.invoke(cli, ["status"])

                # Should fail due to missing API key
                assert result.exit_code != 0
                assert result.exception is not None

    def test_index_with_clear(self, runner, mock_ragzoom):
        """Test indexing with --clear option."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for clear.")
            temp_file = f.name

        try:
            # Mock existing document
            mock_session = MagicMock()
            mock_doc = MagicMock()
            mock_doc.id = os.path.basename(temp_file)
            mock_session.query.return_value.filter_by.return_value.first.return_value = (
                mock_doc
            )
            mock_session.query.return_value.filter_by.return_value.delete.return_value = (
                1
            )

            mock_ragzoom[
                "store_instance"
            ].SessionLocal.return_value.__enter__.return_value = mock_session
            mock_ragzoom["store_instance"].delete_document_nodes.return_value = 5

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file, "--clear"])

                assert result.exit_code == 0
                assert (
                    f"Clearing existing document '{os.path.basename(temp_file)}'"
                    in result.output
                )
                assert "Cleared 5 nodes" in result.output
                assert "Document indexed successfully!" in result.output

                # Verify delete_document_nodes was called
                mock_ragzoom[
                    "store_instance"
                ].delete_document_nodes.assert_called_once_with(
                    os.path.basename(temp_file)
                )
        finally:
            os.unlink(temp_file)

    def test_index_with_clear_no_existing(self, runner, mock_ragzoom):
        """Test indexing with --clear when document doesn't exist."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            # Mock no existing document
            mock_session = MagicMock()
            mock_session.query.return_value.filter_by.return_value.first.return_value = (
                None
            )

            mock_ragzoom[
                "store_instance"
            ].SessionLocal.return_value.__enter__.return_value = mock_session

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file, "--clear"])

                assert result.exit_code == 0
                # Should not show clearing message
                assert "Clearing existing document" not in result.output
                assert "Document indexed successfully!" in result.output

                # Verify delete_document_nodes was NOT called
                mock_ragzoom["store_instance"].delete_document_nodes.assert_not_called()
        finally:
            os.unlink(temp_file)

    def test_index_no_progress(self, runner, mock_ragzoom):
        """Test indexing without progress bar."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file, "--no-progress"])

                assert result.exit_code == 0

                # Verify add_document was called with show_progress=False
                call_args = mock_ragzoom["builder_instance"].add_document.call_args
                assert call_args[1]["show_progress"] is False
        finally:
            os.unlink(temp_file)

    def test_clear_specific_document(self, runner, mock_ragzoom):
        """Test clearing a specific document."""
        # Mock document exists
        mock_session = MagicMock()
        mock_doc = MagicMock()
        mock_doc.id = "test-doc"
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            mock_doc
        )
        mock_session.query.return_value.filter_by.return_value.delete.return_value = 1

        mock_ragzoom[
            "store_instance"
        ].SessionLocal.return_value.__enter__.return_value = mock_session
        mock_ragzoom["store_instance"].delete_document_nodes.return_value = 10

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli, ["clear", "--document-id", "test-doc", "--confirm"]
            )

            assert result.exit_code == 0
            assert "Cleared document 'test-doc' (10 nodes deleted)" in result.output

            # Verify delete_document_nodes was called
            mock_ragzoom[
                "store_instance"
            ].delete_document_nodes.assert_called_once_with("test-doc")

    def test_clear_document_not_found(self, runner, mock_ragzoom):
        """Test clearing a non-existent document."""
        # Mock document doesn't exist
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        mock_ragzoom[
            "store_instance"
        ].SessionLocal.return_value.__enter__.return_value = mock_session

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli, ["clear", "--document-id", "non-existent", "--confirm"]
            )

            assert result.exit_code == 1
            assert "Document 'non-existent' not found" in result.output

    def test_clear_all_data(self, runner, mock_ragzoom):
        """Test clearing all data."""
        # Mock database state
        mock_session = MagicMock()
        mock_session.query.return_value.count.return_value = 50

        mock_ragzoom[
            "store_instance"
        ].SessionLocal.return_value.__enter__.return_value = mock_session
        mock_ragzoom["store_instance"].collection.get.return_value = {
            "ids": ["id1", "id2"]
        }

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["clear", "--confirm"])

            assert result.exit_code == 0
            assert "Cleared 50 nodes from the database" in result.output
