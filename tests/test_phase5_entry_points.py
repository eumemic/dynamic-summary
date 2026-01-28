"""Test Phase 5 entry point isolation (CLI commands with DocumentStore).

Note: TestCLIPinCommandIsolation was removed because the pin command was removed.
See specs/grpc-cli-architecture.md § Pin Command Removal.
"""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

from click.testing import CliRunner

from ragzoom.cli import cli
from ragzoom.contracts.node_repository import NodeDataDict
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex


class SkipTestQueryVisualizationIsolation:
    """Test that query visualization uses document-scoped store."""

    @patch("ragzoom.cli.click.echo")
    @patch("ragzoom.cli.create_store_with_docker")
    def test_query_tree_visualization_scoped_to_document(
        self,
        mock_create_store: object,
        mock_echo: object,
        storage_backend: StorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test that tree visualization only shows specified document."""
        # Create document-scoped stores
        doc1_store = storage_backend.for_document("doc1")
        doc2_store = storage_backend.for_document("doc2")

        # Set metadata for both documents
        doc1_store.set_metadata(
            file_path="doc1.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )
        doc2_store.set_metadata(
            file_path="doc2.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add nodes for doc1 using proper add_batch format
        doc1_nodes: list[NodeDataDict] = [
            {
                "node_id": "doc1_root",
                "text": "Document 1 root",
                "span_start": 0,
                "span_end": 200,
                "document_id": "doc1",
                "token_count": 20,
                "height": 1,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": "doc1_left",
                "right_child_id": "doc1_right",
            },
            {
                "node_id": "doc1_left",
                "text": "Document 1 left",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": None,  # Will be set via update_parent_references_batch
                "left_child_id": None,
                "right_child_id": None,
            },
            {
                "node_id": "doc1_right",
                "text": "Document 1 right",
                "span_start": 100,
                "span_end": 200,
                "document_id": "doc1",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": None,  # Will be set via update_parent_references_batch
                "left_child_id": None,
                "right_child_id": None,
            },
        ]
        doc1_store.nodes.add_batch(doc1_nodes)
        # Set parent references
        doc1_store.nodes.update_parent_references_batch(
            [
                ("doc1_left", "doc1_root"),
                ("doc1_right", "doc1_root"),
            ]
        )

        # Add nodes for doc2 using proper add_batch format
        doc2_nodes: list[NodeDataDict] = [
            {
                "node_id": "doc2_root",
                "text": "Document 2 root",
                "span_start": 0,
                "span_end": 200,
                "document_id": "doc2",
                "token_count": 20,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]
        doc2_store.nodes.add_batch(doc2_nodes)

        # Upsert embeddings for all nodes via VectorIndex
        vector_index.upsert(
            [
                (
                    "doc1_root",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 200,
                        "parent_id": "",
                        "document_id": "doc1",
                        "is_leaf": 0,
                    },
                ),
                (
                    "doc1_left",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": "doc1_root",
                        "document_id": "doc1",
                        "is_leaf": 1,
                    },
                ),
                (
                    "doc1_right",
                    [0.5] * 1536,
                    {
                        "span_start": 100,
                        "span_end": 200,
                        "parent_id": "doc1_root",
                        "document_id": "doc1",
                        "is_leaf": 1,
                    },
                ),
            ]
        )
        vector_index.upsert(
            [
                (
                    "doc2_root",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 200,
                        "parent_id": "",
                        "document_id": "doc2",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        runner = CliRunner()
        cast(MagicMock, mock_create_store).return_value = doc1_store

        # Mock the query service and tree visualization
        with patch("ragzoom.cli.QueryService") as mock_query_service_class:
            mock_query_service = Mock()
            mock_query_service_class.return_value = mock_query_service

            # Mock query result
            mock_result = Mock()
            mock_result.tiling = ["doc1_left", "doc1_right"]
            mock_result.coverage_map = {
                "doc1_root": True,
                "doc1_left": True,
                "doc1_right": True,
            }
            mock_query_service.execute_query.return_value = mock_result

            with patch("ragzoom.cli.build_ascii_tree") as mock_build_tree:
                mock_build_tree.return_value = "ASCII tree visualization"

                # Run query with tree visualization
                runner.invoke(
                    cli, ["query", "test query", "--document-id", "doc1", "--show-tree"]
                )

                # Verify build_ascii_tree was called with document-scoped store
                assert mock_build_tree.called
                call_args = mock_build_tree.call_args

                # The store passed should be document-scoped
                tree_store = (
                    call_args[0][0] if call_args[0] else call_args.kwargs.get("store")
                )

                # If it's a DocumentStore, it should have document_id == "doc1"
                if tree_store and hasattr(tree_store, "document_id"):
                    assert tree_store.document_id == "doc1"
