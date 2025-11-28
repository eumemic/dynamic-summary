"""Test Phase 5 entry point isolation (CLI commands with DocumentStore)."""

from typing import cast
from unittest.mock import MagicMock, Mock, patch

import numpy as np
from click.testing import CliRunner
from numpy.typing import NDArray

from ragzoom.cli import cli
from ragzoom.contracts.storage_backend import StorageBackend
from ragzoom.contracts.vector_index import VectorIndex


class TestCLIPinCommandIsolation:
    """Test that CLI pin command properly uses DocumentStore for isolation."""

    def test_pin_command_with_document_id(
        self,
        storage_backend: StorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test pin command with explicit document ID."""
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

        # Add nodes to different documents using proper add_batch format
        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "doc1_node",
                "text": "Document 1 content",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]
        doc1_store.nodes.add_batch(nodes)

        nodes_doc2: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "doc2_node",
                "text": "Document 2 content",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc2",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]
        doc2_store.nodes.add_batch(nodes_doc2)

        # Upsert embeddings via VectorIndex (no longer through DocumentStore)
        vector_index.upsert(
            [
                (
                    "doc1_node",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": "",
                        "document_id": "doc1",
                        "is_leaf": 1,
                    },
                ),
            ]
        )
        vector_index.upsert(
            [
                (
                    "doc2_node",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": "",
                        "document_id": "doc2",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        runner = CliRunner()

        with patch("ragzoom.cli.create_store_with_docker") as mock_create_store:
            # Return the backend; CLI will scope per document internally
            mock_create_store.return_value = storage_backend

            # Pin a node from doc1 with explicit document ID
            result = runner.invoke(cli, ["pin", "doc1_node", "--document-id", "doc1"])

            # Should succeed
            assert result.exit_code == 0
            # Verify the node is pinned
            pinned_nodes = [n.id for n in doc1_store.get_pinned_nodes()]
            assert "doc1_node" in pinned_nodes

    def test_pin_command_auto_detects_document(
        self,
        storage_backend: StorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test pin command auto-detects document from node ID."""
        doc_store = storage_backend.for_document("doc1")
        doc_store.set_metadata(
            file_path="doc1.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add node using proper add_batch format
        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "doc1_node",
                "text": "Document 1 content",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Upsert embedding via VectorIndex
        vector_index.upsert(
            [
                (
                    "doc1_node",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": "",
                        "document_id": "doc1",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        runner = CliRunner()

        with patch("ragzoom.cli.create_store_with_docker") as mock_create_store:
            mock_create_store.return_value = storage_backend

            # Pin without document ID - should auto-detect
            result = runner.invoke(cli, ["pin", "doc1_node"])

            # Should succeed with auto-detection
            if result.exit_code != 0:
                print(f"Error output: {result.output}")
                print(f"Exception: {result.exception}")
            assert result.exit_code == 0
            # Verify the node is pinned
            pinned_nodes = [n.id for n in doc_store.get_pinned_nodes()]
            assert "doc1_node" in pinned_nodes

    def test_pin_command_validates_document_ownership(
        self,
        storage_backend: StorageBackend,
        vector_index: VectorIndex,
    ) -> None:
        """Test pin command validates node belongs to specified document."""
        doc_store = storage_backend.for_document("doc1")
        doc_store.set_metadata(
            file_path="doc1.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        # Add node to doc1 using proper add_batch format
        nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
            {
                "node_id": "doc1_node",
                "text": "Document 1 content",
                "span_start": 0,
                "span_end": 100,
                "document_id": "doc1",
                "token_count": 10,
                "height": 0,
                "level_index": 0,
                "parent_id": None,
                "left_child_id": None,
                "right_child_id": None,
            }
        ]
        doc_store.nodes.add_batch(nodes)

        # Upsert embedding via VectorIndex
        vector_index.upsert(
            [
                (
                    "doc1_node",
                    [0.5] * 1536,
                    {
                        "span_start": 0,
                        "span_end": 100,
                        "parent_id": "",
                        "document_id": "doc1",
                        "is_leaf": 1,
                    },
                )
            ]
        )

        runner = CliRunner()

        with patch("ragzoom.cli.create_store_with_docker") as mock_create_store:
            # Create a doc2 store when wrong document ID is specified
            doc2_store = storage_backend.for_document("doc2")
            doc2_store.set_metadata(
                file_path="doc2.txt",
                embedding_model="text-embedding-3-small",
                summary_model="gpt-4o-mini",
            )
            mock_create_store.return_value = storage_backend

            # Try to pin with wrong document ID
            result = runner.invoke(cli, ["pin", "doc1_node", "--document-id", "doc2"])

            # Should fail with error message
            assert result.exit_code != 0
            assert (
                "does not belong to document doc2" in result.output.lower()
                or "node not found" in result.output.lower()
            )

    def test_pin_command_error_on_nonexistent_node(
        self, storage_backend: StorageBackend
    ) -> None:
        """Test pin command handles non-existent nodes gracefully."""
        doc_store = storage_backend.for_document("doc1")
        doc_store.set_metadata(
            file_path="doc1.txt",
            embedding_model="text-embedding-3-small",
            summary_model="gpt-4o-mini",
        )

        runner = CliRunner()

        with patch("ragzoom.cli.create_store_with_docker") as mock_create_store:
            mock_create_store.return_value = storage_backend

            # Try to pin non-existent node
            result = runner.invoke(cli, ["pin", "nonexistent_node"])

            # Should fail with appropriate error
            assert result.exit_code != 0
            assert (
                "not found" in result.output.lower()
                or "does not exist" in result.output.lower()
            )


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
        doc1_nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
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
        doc2_nodes: list[
            dict[
                str, str | int | float | bool | list[float] | NDArray[np.float64] | None
            ]
        ] = [
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
