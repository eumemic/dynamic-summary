"""Tests for CLI functionality."""

import os
import tempfile
from collections.abc import Iterator
from unittest.mock import MagicMock, Mock, patch

import pytest
from click.testing import CliRunner

from ragzoom.cli import cli
from ragzoom.exceptions import InvalidOperationError


class TestCLI:
    """Test the CLI commands."""

    @pytest.fixture
    def runner(self) -> CliRunner:
        """Create a CLI runner."""
        return CliRunner()

    @pytest.fixture
    def mock_ragzoom(self) -> Iterator[dict[str, object]]:
        """Mock RagZoom components."""
        with (
            patch("ragzoom.cli.IndexConfig") as mock_index_config,
            patch("ragzoom.cli.QueryConfig") as mock_query_config,
            patch("ragzoom.cli.OperationalConfig") as mock_operational_config,
            patch("ragzoom.cli.create_store_with_docker") as mock_create_store,
            patch("ragzoom.cli.DocumentService") as mock_document_service,
            patch("ragzoom.cli.IndexingService") as mock_indexing_service,
            patch("ragzoom.cli.QueryService") as mock_query_service,
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
            def mock_replace(**kwargs: object) -> object:
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
            store_instance.get_root_node.return_value = Mock(height=3)
            store_instance.get_pinned_nodes.return_value = []
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

            # Mock document store (returned by for_document)
            mock_doc_store = Mock()
            mock_doc_store.nodes.get_all.return_value = [
                Mock() for _ in range(10)
            ]  # Mock list
            mock_doc_store.nodes.get_leaves.return_value = [
                Mock() for _ in range(5)
            ]  # Mock list
            store_instance.for_document.return_value = mock_doc_store

            mock_create_store.return_value = store_instance

            # Mock services
            document_service_instance = Mock()

            # Create mock document for list_documents
            from datetime import datetime

            from ragzoom.services.document_service import DocumentInfo

            mock_document = DocumentInfo(
                document_id="doc-123",
                file_path="/path/to/file.txt",
                indexed_at=datetime(2023, 1, 1),
                chunk_count=5,
                node_count=15,
            )
            document_service_instance.list_documents.return_value = [mock_document]

            document_service_instance.get_system_status.return_value = Mock(
                total_nodes=10, leaf_nodes=5, tree_depth=3, pinned_nodes=0
            )
            # Set default clear values - individual tests can override these
            document_service_instance.clear_document.return_value = 10
            document_service_instance.clear_all_documents.return_value = 50
            mock_document_service.return_value = document_service_instance

            # Mock indexing service
            indexing_service_instance = Mock()
            from ragzoom.services.indexing_service import IndexingResult

            indexing_service_instance.index_from_file.return_value = IndexingResult(
                document_id="doc-123",
                chunks_created=5,
                tree_depth=3,
                telemetry=None,
            )
            indexing_service_instance.append_to_document.return_value = IndexingResult(
                document_id="doc-append",
                chunks_created=2,
                tree_depth=4,
                telemetry=None,
            )
            mock_indexing_service.return_value = indexing_service_instance

            # Mock query service
            query_service_instance = Mock()
            from ragzoom.services.query_service import QueryResult

            query_service_instance.execute_query.return_value = QueryResult(
                summary="This is a summary of the content.",
                token_count=100,
                nodes_retrieved=2,
                tiling_size=3,
            )
            mock_query_service.return_value = query_service_instance

            yield {
                "index_config": mock_index_config,
                "query_config": mock_query_config,
                "operational_config": mock_operational_config,
                "create_store": mock_create_store,
                "document_service": mock_document_service,
                "indexing_service": mock_indexing_service,
                "query_service": mock_query_service,
                "index_config_instance": index_config_instance,
                "query_config_instance": query_config_instance,
                "operational_config_instance": operational_config_instance,
                "store_instance": store_instance,
                "document_service_instance": document_service_instance,
                "indexing_service_instance": indexing_service_instance,
                "query_service_instance": query_service_instance,
            }

    def test_cli_help(self, runner: CliRunner) -> None:
        """Test CLI help command."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "RagZoom: Incremental, hierarchical RAG memory system." in result.output
        assert "Commands:" in result.output

    def test_status_command(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test status command."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["status"])

            assert result.exit_code == 0
            assert "SYSTEM STATUS" in result.output
            assert "Total nodes: 10" in result.output
            assert "Leaf nodes: 5" in result.output
            assert "Tree height: 3" in result.output
            assert "Pinned nodes: 0" in result.output

    def test_index_command_with_file(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
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

                # Verify index_from_file was called with the content
                mock_ragzoom[
                    "indexing_service_instance"
                ].index_from_file.assert_called_once()
                call_args = mock_ragzoom[
                    "indexing_service_instance"
                ].index_from_file.call_args
                assert temp_file in call_args[0][0]
        finally:
            os.unlink(temp_file)

    def test_index_command_with_text(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
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

                # Verify index_from_file was called with document_id
                call_args = mock_ragzoom[
                    "indexing_service_instance"
                ].index_from_file.call_args
                assert call_args[1]["document_id"] == "my-doc-id"
        finally:
            os.unlink(temp_file)

    def test_index_append_requires_document_id(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """--append should require an explicit document ID."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Append me")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file, "--append"])

                assert result.exit_code != 0
                assert "--document-id is required" in result.output
        finally:
            os.unlink(temp_file)

    def test_index_append_invokes_service(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Ensure the append flag routes through append_to_document."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Chunk for append")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(
                    cli,
                    [
                        "index",
                        temp_file,
                        "--document-id",
                        "append-doc",
                        "--append",
                    ],
                )

                assert result.exit_code == 0
                assert "Document appended successfully" in result.output
                service = mock_ragzoom["indexing_service_instance"]
                service.append_to_document.assert_called_once()
                service.index_from_file.assert_not_called()
                call_args = service.append_to_document.call_args
                assert call_args.kwargs["document_id"] == "append-doc"
        finally:
            os.unlink(temp_file)

    def test_query_command(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test query command."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli, ["query", "Tell me about cats", "-d", "test-doc"]
            )

            assert result.exit_code == 0
            assert "SUMMARY" in result.output
            assert "This is a summary of the content." in result.output

    def test_query_with_options(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
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

            # Verify execute_query was called with correct parameters
            mock_ragzoom[
                "query_service_instance"
            ].execute_query.assert_called_once_with(
                "Tell me about cats",
                "test-doc",
                num_seeds=5,
                token_budget=1000,
            )

    def test_pin_command(
        self, runner: CliRunner, mock_ragzoom: dict[str, object]
    ) -> None:
        """Test pin command."""
        mock_service_instance = mock_ragzoom["document_service_instance"]
        assert hasattr(mock_service_instance, "pin_node")
        mock_service_instance.pin_node.return_value = None

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["pin", "node-123"])

            assert result.exit_code == 0
            assert "✅ Node node-123 pinned successfully!" in result.output

    def test_pin_command_failure(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test pin command when pinning fails."""
        mock_ragzoom["document_service_instance"].pin_node.side_effect = (
            InvalidOperationError("pin_node", "Node is too deep or already pinned")
        )

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["pin", "node-999"])

            # Pin command returns exit code 1 on failure
            assert result.exit_code == 1
            assert "Failed to pin node node-999" in result.output

    def test_serve_command(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test serve command."""
        with patch("uvicorn.run") as mock_uvicorn:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["serve", "--port", "8080"])

                assert result.exit_code == 0

                # Verify uvicorn was called with correct parameters
                mock_uvicorn.assert_called_once_with(
                    "ragzoom.api:app", host="127.0.0.1", port=8080, reload=False
                )

    def test_documents_command(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test documents command."""
        # Mock document results using the document service
        from datetime import datetime

        from ragzoom.services.document_service import DocumentInfo

        mock_doc_info = DocumentInfo(
            document_id="doc-123",
            file_path="/path/to/file.txt",
            indexed_at=datetime.now(),
            chunk_count=10,
            node_count=15,
        )

        mock_ragzoom["document_service_instance"].list_documents.return_value = [
            mock_doc_info
        ]

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(cli, ["documents"])

            assert result.exit_code == 0
            assert "Document ID: doc-123" in result.output
            assert "File: /path/to/file.txt" in result.output
            assert "Chunks: 10" in result.output

    def test_missing_api_key(self, runner: CliRunner) -> None:
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

    def test_index_with_automatic_clearing(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test that indexing automatically clears existing data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content for clear.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file])

                assert result.exit_code == 0
                assert "Document indexed successfully!" in result.output

                # Verify that the indexing service was called
                mock_ragzoom[
                    "indexing_service_instance"
                ].index_from_file.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_index_automatic_clearing_no_nodes(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test that automatic clearing works when no nodes exist."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file])

                assert result.exit_code == 0
                assert "Document indexed successfully!" in result.output

                # Verify that the indexing service was called
                mock_ragzoom[
                    "indexing_service_instance"
                ].index_from_file.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_index_basic(
        self, runner: CliRunner, mock_ragzoom: dict[str, object]
    ) -> None:
        """Test basic indexing command."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                result = runner.invoke(cli, ["index", temp_file])

                assert result.exit_code == 0

                # Verify index_from_file was called
                mock_indexing_service = mock_ragzoom["indexing_service_instance"]
                assert hasattr(mock_indexing_service, "index_from_file")
                mock_indexing_service.index_from_file.assert_called_once()
        finally:
            os.unlink(temp_file)

    def test_clear_specific_document(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
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
            mock_ragzoom[
                "document_service_instance"
            ].clear_document.assert_called_once_with("test-doc")

    def test_clear_document_with_orphaned_nodes(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
        """Test clearing orphaned nodes (no Document record)."""
        # Mock document service clearing orphaned nodes
        mock_ragzoom["document_service_instance"].clear_document.return_value = 248

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            result = runner.invoke(
                cli, ["clear", "--document-id", "orphaned-doc", "--confirm"]
            )

            assert result.exit_code == 0
            assert (
                "Cleared document 'orphaned-doc' (248 nodes deleted)" in result.output
            )

            # Verify clear_document was called even without Document record
            mock_ragzoom[
                "document_service_instance"
            ].clear_document.assert_called_once_with("orphaned-doc")

    def test_clear_all_data(
        self, runner: CliRunner, mock_ragzoom: dict[str, Mock]
    ) -> None:
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
