"""Tests for CLI functionality."""

import os
import tempfile
from unittest.mock import Mock, patch

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
        with patch('ragzoom.cli.RagZoomConfig') as mock_config, \
             patch('ragzoom.cli.Store') as mock_store, \
             patch('ragzoom.index.TreeBuilder') as mock_builder, \
             patch('ragzoom.cli.Retriever') as mock_retriever, \
             patch('ragzoom.cli.Assembler') as mock_assembler:

            # Mock config
            config_instance = Mock()
            config_instance.budget_tokens = 8000
            config_instance.mmr_lambda = 0.7
            mock_config.return_value = config_instance

            # Mock store
            store_instance = Mock()
            store_instance.get_leaf_nodes.return_value = [Mock(id=f"node-{i}") for i in range(5)]
            store_instance.get_root_node.return_value = Mock(depth=3)
            store_instance.get_pinned_nodes.return_value = []
            store_instance.collection.count.return_value = 10
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
                coverage_map={"node-1": 1.0, "node-2": 1.0}
            )
            mock_retriever.return_value = retriever_instance

            # Mock assembler
            assembler_instance = Mock()
            assembler_instance.assemble.return_value = "This is a summary of the content."
            assembler_instance.assemble_with_budget.return_value = ("This is a summary of the content.", 100)
            mock_assembler.return_value = assembler_instance

            yield {
                'config': mock_config,
                'store': mock_store,
                'builder': mock_builder,
                'retriever': mock_retriever,
                'assembler': mock_assembler,
                'config_instance': config_instance,
                'store_instance': store_instance,
                'builder_instance': builder_instance,
                'retriever_instance': retriever_instance,
                'assembler_instance': assembler_instance
            }

    def test_cli_help(self, runner):
        """Test CLI help command."""
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'RagZoom: Incremental, hierarchical RAG memory system.' in result.output
        assert 'Commands:' in result.output

    def test_status_command(self, runner, mock_ragzoom):
        """Test status command."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = runner.invoke(cli, ['status'])

            assert result.exit_code == 0
            assert 'SYSTEM STATUS' in result.output
            assert 'Total nodes: 10' in result.output
            assert 'Leaf nodes: 5' in result.output
            assert 'Tree depth: 3' in result.output

    def test_index_command_with_file(self, runner, mock_ragzoom):
        """Test indexing a file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content for indexing.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                result = runner.invoke(cli, ['index', temp_file])

                assert result.exit_code == 0
                assert 'Document indexed successfully!' in result.output
                assert 'doc-123' in result.output

                # Verify add_document was called with the content
                mock_ragzoom['builder_instance'].add_document.assert_called_once()
                call_args = mock_ragzoom['builder_instance'].add_document.call_args
                assert "Test content for indexing." in call_args[0][0]
        finally:
            os.unlink(temp_file)

    def test_index_command_with_text(self, runner, mock_ragzoom):
        """Test indexing with document ID."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                result = runner.invoke(cli, ['index', temp_file, '--document-id', 'my-doc-id'])

                assert result.exit_code == 0
                assert 'Document indexed successfully!' in result.output
                assert 'doc-123' in result.output

                # Verify add_document was called with document_id
                call_args = mock_ragzoom['builder_instance'].add_document.call_args
                assert call_args[1]['document_id'] == 'my-doc-id'
        finally:
            os.unlink(temp_file)

    def test_query_command(self, runner, mock_ragzoom):
        """Test query command."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = runner.invoke(cli, ['query', 'Tell me about cats'])

            assert result.exit_code == 0
            assert 'SUMMARY:' in result.output
            assert 'This is a summary of the content.' in result.output

    def test_query_with_options(self, runner, mock_ragzoom):
        """Test query command with options."""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = runner.invoke(cli, ['query', 'Tell me about cats',
                                        '--n-max', '5',
                                        '--token-budget', '1000'])

            assert result.exit_code == 0

            # Verify retrieve was called with correct query, n_max, and budget_tokens
            mock_ragzoom['retriever_instance'].retrieve.assert_called_once_with(
                'Tell me about cats', n_max=5, budget_tokens=1000
            )

    def test_pin_command(self, runner, mock_ragzoom):
        """Test pin command."""
        mock_ragzoom['store_instance'].pin_node.return_value = True

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = runner.invoke(cli, ['pin', 'node-123'])

            assert result.exit_code == 0
            assert '✅ Node node-123 pinned successfully!' in result.output

    def test_pin_command_failure(self, runner, mock_ragzoom):
        """Test pin command when pinning fails."""
        mock_ragzoom['store_instance'].pin_node.return_value = False

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = runner.invoke(cli, ['pin', 'node-999'])

            # Pin command returns exit code 1 on failure
            assert result.exit_code == 1
            assert 'Failed to pin node node-999' in result.output


    def test_serve_command(self, runner, mock_ragzoom):
        """Test serve command."""
        with patch('uvicorn.run') as mock_uvicorn:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                result = runner.invoke(cli, ['serve', '--port', '8080'])

                assert result.exit_code == 0

                # Verify uvicorn was called with correct parameters
                mock_uvicorn.assert_called_once_with(
                    "ragzoom.api:app",
                    host="0.0.0.0",
                    port=8080,
                    reload=False
                )

    def test_missing_api_key(self, runner):
        """Test commands fail without API key."""
        # Mock the RagZoomConfig to raise an error when instantiated without API key
        with patch('ragzoom.cli.RagZoomConfig') as mock_config:
            mock_config.side_effect = ValueError("Field required: openai_api_key")

            with patch.dict(os.environ, {'OPENAI_API_KEY': ''}, clear=True):
                result = runner.invoke(cli, ['status'])

                # Should fail due to missing API key
                assert result.exit_code != 0
                assert result.exception is not None

    def test_index_no_progress(self, runner, mock_ragzoom):
        """Test indexing without progress bar."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test content.")
            temp_file = f.name

        try:
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
                result = runner.invoke(cli, ['index', temp_file, '--no-progress'])

                assert result.exit_code == 0

                # Verify add_document was called with show_progress=False
                call_args = mock_ragzoom['builder_instance'].add_document.call_args
                assert call_args[1]['show_progress'] is False
        finally:
            os.unlink(temp_file)

    def test_query_with_eviction(self, runner, mock_ragzoom):
        """Test query with eviction mode."""
        mock_ragzoom['retriever_instance'].retrieve_with_eviction.return_value = Mock(
            node_ids=["node-1"],
            frontier_nodes=["node-1"],
            coverage_map={"node-1": 1.0}
        )

        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-key'}):
            result = runner.invoke(cli, ['query', 'Test query', '--use-eviction'])

            assert result.exit_code == 0

            # Verify retrieve_with_eviction was called
            mock_ragzoom['retriever_instance'].retrieve_with_eviction.assert_called_once()

            assert 'SUMMARY:' in result.output
