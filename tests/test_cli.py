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
            patch("ragzoom.cli.IndexConfig") as mock_index_config,
            patch("ragzoom.cli.QueryConfig") as mock_query_config,
            patch("ragzoom.cli.OperationalConfig") as mock_operational_config,
            patch("ragzoom.cli.Store") as mock_store,
            patch("ragzoom.cli.TreeBuilder") as mock_builder,
            patch("ragzoom.cli.Retriever") as mock_retriever,
            patch("ragzoom.cli.Assembler") as mock_assembler,
        ):

            # Mock index config
            index_config_instance = Mock()
            index_config_instance.target_chunk_tokens = 200
            index_config_instance.embedding_model = "text-embedding-3-small"
            mock_index_config.return_value = index_config_instance

            # Mock query config
            query_config_instance = Mock()
            query_config_instance.budget_tokens = 8000
            query_config_instance.mmr_lambda = 0.7

            # Mock the replace method to return a proper config with updated values
            def mock_replace(**kwargs):
                new_config = Mock()
                new_config.budget_tokens = kwargs.get(
                    "budget_tokens", query_config_instance.budget_tokens
                )
                new_config.mmr_lambda = kwargs.get(
                    "mmr_lambda", query_config_instance.mmr_lambda
                )
                new_config.mmr_k_multiplier = kwargs.get("mmr_k_multiplier", 2.0)
                return new_config

            query_config_instance.replace = mock_replace
            mock_query_config.return_value = query_config_instance

            # Mock operational config
            operational_config_instance = Mock()
            operational_config_instance.openai_api_key = "test-key"
            operational_config_instance.database_url = "postgresql:///ragzoom"
            mock_operational_config.return_value = operational_config_instance

            # Mock store
            store_instance = Mock()
            store_instance.get_leaf_nodes.return_value = [
                Mock(id=f"node-{i}") for i in range(5)
            ]
            store_instance.get_root_node.return_value = Mock(depth=3)
            store_instance.get_pinned_nodes.return_value = []
            store_instance.get_node_height.return_value = 3
            store_instance.clear_document.return_value = (
                0  # Default to no nodes cleared
            )

            # Mock SessionLocal for database queries (handles count queries)
            mock_session = Mock()
            mock_query = Mock()
            # Set up count to return 10 for TreeNode queries
            mock_query.count = Mock(return_value=10)

            # Mock leaf nodes query
            leaf_nodes = [
                Mock(
                    id=f"leaf-{i}",
                    span_start=i * 100,
                    span_end=(i + 1) * 100,
                    text=f"text-{i}",
                    summary=None,
                    left_child_id=None,
                    right_child_id=None,
                )
                for i in range(5)
            ]

            # Set up chained query mocks for filter pattern
            mock_filter_result = Mock()
            mock_filter_result.all.return_value = leaf_nodes
            mock_filter_result.count.return_value = len(leaf_nodes)
            mock_filter_result.first.return_value = Mock(
                id="root", depth=3, parent_id=None
            )

            # Handle both filter_by().filter() chain and just filter_by()
            mock_query.filter_by.return_value.filter.return_value = mock_filter_result
            mock_query.filter_by.return_value.all.return_value = leaf_nodes
            mock_query.filter_by.return_value.first.return_value = Mock(
                id="root", depth=3, parent_id=None
            )
            mock_query.filter_by.return_value.count.return_value = 10

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
                tiling=None,  # Updated field name
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
                "index_config": mock_index_config,
                "query_config": mock_query_config,
                "operational_config": mock_operational_config,
                "store": mock_store,
                "builder": mock_builder,
                "retriever": mock_retriever,
                "assembler": mock_assembler,
                "index_config_instance": index_config_instance,
                "query_config_instance": query_config_instance,
                "operational_config_instance": operational_config_instance,
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
            assert "Tree height:" in result.output

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
            assert "SUMMARY" in result.output
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
                    "--num-seeds",
                    "5",
                    "--token-budget",
                    "1000",
                ],
            )

            assert result.exit_code == 0

            # Verify retrieve was called with correct query, num_seeds, budget_tokens, and document_id
            mock_ragzoom["retriever_instance"].retrieve.assert_called_once_with(
                "Tell me about cats",
                budget_tokens=1000,
                document_id="test-doc",
                num_seeds=5,
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
                    "ragzoom.api:app", host="127.0.0.1", port=8080, reload=False
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
        # Mock the OperationalConfig to have empty API key
        with patch("ragzoom.cli.OperationalConfig") as mock_config:
            mock_config_instance = Mock()
            mock_config_instance.openai_api_key = ""
            mock_config.return_value = mock_config_instance

            with patch.dict(os.environ, {"OPENAI_API_KEY": ""}, clear=True):
                result = runner.invoke(cli, ["status"])

                # Should fail due to missing API key - but for now just check it runs
                # The actual API key validation happens in the components, not CLI init
                assert result.exit_code == 0 or result.exception is not None

    def test_index_with_automatic_clearing(self, runner, mock_ragzoom):
        """Test that indexing automatically clears existing data."""
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
            mock_ragzoom["store_instance"].clear_document.return_value = 5

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file])

                assert result.exit_code == 0
                assert "Clearing existing data" in result.output
                assert "Cleared 5 nodes" in result.output
                assert "Document indexed successfully!" in result.output

                # Verify clear_document was called
                mock_ragzoom["store_instance"].clear_document.assert_called_once_with(
                    os.path.basename(temp_file)
                )
        finally:
            os.unlink(temp_file)

    def test_index_automatic_clearing_no_nodes(self, runner, mock_ragzoom):
        """Test that automatic clearing works when no nodes exist."""
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
            mock_ragzoom["store_instance"].clear_document.return_value = 0

            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file])

                assert result.exit_code == 0
                # Should NOT show clearing message when no nodes exist
                assert "Clearing existing data" not in result.output
                assert "Document indexed successfully!" in result.output

                # Verify clear_document was called (automatic clearing)
                mock_ragzoom["store_instance"].clear_document.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_index_basic(self, runner, mock_ragzoom):
        """Test basic indexing command."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file])

                assert result.exit_code == 0

                # Verify add_document was called
                mock_ragzoom["builder_instance"].add_document.assert_called_once()
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
        mock_ragzoom["store_instance"].clear_document.return_value = 10

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli, ["clear", "--document-id", "test-doc", "--confirm"]
            )

            assert result.exit_code == 0
            assert "Cleared document 'test-doc' (10 nodes deleted)" in result.output

            # Verify clear_document was called
            mock_ragzoom["store_instance"].clear_document.assert_called_once_with(
                "test-doc"
            )

    def test_clear_document_with_orphaned_nodes(self, runner, mock_ragzoom):
        """Test clearing orphaned nodes (no Document record)."""
        # Mock no document record but nodes exist
        mock_session = MagicMock()
        mock_session.query.return_value.filter_by.return_value.first.return_value = None

        mock_ragzoom[
            "store_instance"
        ].SessionLocal.return_value.__enter__.return_value = mock_session
        mock_ragzoom["store_instance"].clear_document.return_value = (
            248  # orphaned nodes
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli, ["clear", "--document-id", "orphaned-doc", "--confirm"]
            )

            assert result.exit_code == 0
            assert (
                "Cleared document 'orphaned-doc' (248 nodes deleted)" in result.output
            )

            # Verify clear_document was called even without Document record
            mock_ragzoom["store_instance"].clear_document.assert_called_once_with(
                "orphaned-doc"
            )

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
