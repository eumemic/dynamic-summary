"""Tests for the service layer implementation."""

from datetime import datetime
from unittest.mock import Mock, patch

from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig, SecretStr
from ragzoom.services.document_service import (
    DocumentInfo,
    DocumentService,
    SystemStatus,
)
from ragzoom.services.indexing_service import IndexingResult, IndexingService
from ragzoom.services.query_service import QueryResult, QueryService


class TestDocumentService:
    """Test the DocumentService."""

    def test_list_documents(self):
        """Test listing documents returns formatted results."""
        # Create mock store with documents
        mock_store = Mock()
        mock_session = Mock()
        mock_doc = Mock()
        mock_doc.id = "test-doc"
        mock_doc.file_path = "/path/to/file.txt"
        mock_doc.indexed_at = datetime(2023, 1, 1, 12, 0, 0)
        mock_doc.chunk_count = 5

        # Mock the session query chain
        mock_session.query.return_value.all.return_value = [mock_doc]
        mock_session.query.return_value.filter_by.return_value.count.return_value = 10

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_store.SessionLocal.return_value = mock_context_manager

        # Create service and test
        service = DocumentService(mock_store)
        documents = service.list_documents()

        assert len(documents) == 1
        assert isinstance(documents[0], DocumentInfo)
        assert documents[0].document_id == "test-doc"
        assert documents[0].file_path == "/path/to/file.txt"
        assert documents[0].chunk_count == 5
        assert documents[0].node_count == 10

    def test_get_system_status(self):
        """Test getting system status."""
        mock_store = Mock()
        mock_session = Mock()

        # Create chainable mock for queries
        mock_query = Mock()

        # First query().count() returns total nodes
        mock_query.count.return_value = 100

        # Second query for leaf nodes
        mock_filter = Mock()
        mock_filter.count.return_value = 20
        mock_query.filter.return_value = mock_filter

        # Third query for max height
        mock_order = Mock()
        mock_order.first.return_value = (5,)  # Returns tuple as SQL would
        mock_query.order_by.return_value = mock_order

        # Make query return our mock
        mock_session.query.return_value = mock_query

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_store.SessionLocal.return_value = mock_context_manager

        # Mock pinned nodes
        mock_store.get_pinned_nodes.return_value = [Mock() for _ in range(2)]

        service = DocumentService(mock_store)
        status = service.get_system_status()

        assert isinstance(status, SystemStatus)
        assert status.total_nodes == 100
        assert status.leaf_nodes == 20
        assert status.tree_depth == 5
        assert status.pinned_nodes == 2

    def test_clear_document(self):
        """Test clearing a document."""
        mock_store = Mock()
        mock_store.clear_document.return_value = 15

        service = DocumentService(mock_store)
        deleted_count = service.clear_document("test-doc")

        assert deleted_count == 15
        mock_store.clear_document.assert_called_once_with("test-doc")


class TestIndexingService:
    """Test the IndexingService."""

    @patch("ragzoom.services.indexing_service.TreeBuilder")
    def test_index_document(self, mock_tree_builder_class):
        """Test indexing a document."""
        # Mock dependencies
        mock_store = Mock()
        mock_store.clear_document.return_value = 0

        # Mock TreeBuilder
        mock_tree_builder = Mock()
        mock_tree_builder.add_document.return_value = "doc-123"
        mock_tree_builder_class.return_value = mock_tree_builder

        # Mock database session for stats
        mock_session = Mock()
        mock_session.query.return_value.filter_by.return_value.filter.return_value.all.return_value = [
            Mock() for _ in range(3)
        ]  # 3 leaf nodes
        mock_session.query.return_value.filter_by.return_value.first.return_value = (
            Mock(height=2)
        )

        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_session)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_store.SessionLocal.return_value = mock_context_manager

        # Create configs
        index_config = IndexConfig.load()
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create service and test
        service = IndexingService(mock_store, index_config, operational_config)
        result = service.index_document("test text", document_id="test-doc")

        assert isinstance(result, IndexingResult)
        assert result.document_id == "doc-123"
        assert result.chunks_created == 3
        assert result.tree_depth == 2
        assert result.telemetry is None

        mock_tree_builder.add_document.assert_called_once_with(
            "test text", document_id="test-doc", file_path=None, show_progress=True
        )


class TestQueryService:
    """Test the QueryService."""

    @patch("ragzoom.services.query_service.Retriever")
    @patch("ragzoom.services.query_service.Assembler")
    def test_execute_query(self, mock_assembler_class, mock_retriever_class):
        """Test executing a query."""
        # Mock dependencies
        mock_store = Mock()

        # Mock Retriever
        mock_retriever = Mock()
        mock_retrieval_result = Mock()
        mock_retrieval_result.node_ids = ["node1", "node2"]
        mock_retrieval_result.tiling = ["node1", "node3", "node2"]
        mock_retriever.retrieve.return_value = mock_retrieval_result
        mock_retriever_class.return_value = mock_retriever

        # Mock Assembler
        mock_assembler = Mock()
        mock_assembler.assemble.return_value = "This is the summary"
        mock_assembler.get_token_count.return_value = 50
        mock_assembler_class.return_value = mock_assembler

        # Create configs
        query_config = QueryConfig(budget_tokens=1000)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create service and test
        service = QueryService(mock_store, query_config, operational_config)
        result = service.execute_query("test query", "doc-123")

        assert isinstance(result, QueryResult)
        assert result.summary == "This is the summary"
        assert result.token_count == 50
        assert result.nodes_retrieved == 2
        assert result.tiling_size == 3

        mock_retriever.retrieve.assert_called_once_with(
            "test query", budget_tokens=1000, document_id="doc-123", num_seeds=None
        )
        mock_assembler.assemble.assert_called_once_with(mock_retrieval_result)

    @patch("ragzoom.services.query_service.Retriever")
    def test_update_config(self, mock_retriever_class):
        """Test updating query configuration."""
        mock_store = Mock()

        # Mock original retriever
        mock_original_retriever = Mock()
        mock_retriever_class.return_value = mock_original_retriever

        # Create configs
        query_config = QueryConfig(budget_tokens=1000, mmr_lambda=0.7)
        operational_config = OperationalConfig(openai_api_key=SecretStr("test-key"))

        # Create service
        service = QueryService(mock_store, query_config, operational_config)

        # Mock new retriever for updated config
        mock_new_retriever = Mock()
        mock_retriever_class.return_value = mock_new_retriever

        # Update config
        service.update_config(budget_tokens=2000, mmr_lambda=0.8)

        # Verify config was updated
        assert service.query_config.budget_tokens == 2000
        assert service.query_config.mmr_lambda == 0.8

        # Verify new retriever was created
        assert service.retriever == mock_new_retriever
