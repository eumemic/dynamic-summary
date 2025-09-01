"""Test for ScoringService to ensure it handles numpy array embeddings correctly."""

from unittest.mock import Mock

import numpy as np

from ragzoom.retrieval.scoring_service import ScoringService


class TestScoringService:
    """Test ScoringService functionality."""

    def test_compute_scores_with_numpy_embeddings(self) -> None:
        """Test that ScoringService handles numpy array embeddings correctly.

        This test catches a bug where `if node.embedding:` fails with numpy arrays
        because numpy arrays don't have a simple truth value.
        """
        # Create mock store
        mock_store = Mock()

        # Create service
        service = ScoringService(mock_store)

        # Create mock nodes with numpy array embeddings
        nodes = [
            Mock(
                id="node1",
                embedding=np.array([0.1, 0.2, 0.3]),  # numpy array, not list!
            ),
            Mock(
                id="node2",
                embedding=np.array([0.4, 0.5, 0.6]),  # numpy array, not list!
            ),
            Mock(
                id="node3",
                embedding=None,  # No embedding
            ),
        ]

        # Mock store.nodes.get_nodes to return our nodes
        mock_store.nodes = Mock()
        mock_store.nodes.get_nodes.return_value = nodes

        # Create query embedding
        query_embedding = [0.15, 0.25, 0.35]

        # Call the private method directly that has the bug
        # This method is called by compute_scores for nodes not in initial candidates
        scores: dict[str, float] = {}
        node_ids = {"node1", "node2", "node3"}

        # This should NOT raise ValueError about array truth value
        service._compute_remaining_scores(
            query_embedding=query_embedding,
            node_ids=node_ids,
            scores=scores,
        )

        # Verify results
        assert "node1" in scores
        assert "node2" in scores
        assert "node3" in scores
        assert scores["node3"] == 0.0  # Node without embedding gets 0 score

        # Scores for nodes with embeddings should be non-zero (cosine similarity)
        assert scores["node1"] != 0.0
        assert scores["node2"] != 0.0

    def test_compute_scores_with_empty_numpy_array(self) -> None:
        """Test that empty numpy arrays are handled correctly.

        An empty numpy array should be treated as no embedding.
        """
        # Create mock store
        mock_store = Mock()

        # Create service
        service = ScoringService(mock_store)

        # Create mock node with empty numpy array embedding
        nodes = [
            Mock(
                id="node1",
                embedding=np.array([]),  # Empty numpy array
            ),
        ]

        # Mock store.nodes.get_nodes to return our nodes
        mock_store.nodes = Mock()
        mock_store.nodes.get_nodes.return_value = nodes

        # Create query embedding
        query_embedding = [0.15, 0.25, 0.35]

        # Compute scores - should handle empty array gracefully
        scores: dict[str, float] = {}
        node_ids = {"node1"}

        service._compute_remaining_scores(
            query_embedding=query_embedding,
            node_ids=node_ids,
            scores=scores,
        )

        # Node with empty embedding should get 0 score
        assert scores["node1"] == 0.0

    def test_compute_scores_mixed_embedding_types(self) -> None:
        """Test that ScoringService handles mixed embedding types (lists and arrays).

        In practice, this shouldn't happen, but the service should be robust.
        """
        # Create mock store
        mock_store = Mock()

        # Create service
        service = ScoringService(mock_store)

        # Create mock nodes with mixed embedding types
        nodes = [
            Mock(
                id="node1",
                embedding=[0.1, 0.2, 0.3],  # Python list
            ),
            Mock(
                id="node2",
                embedding=np.array([0.4, 0.5, 0.6]),  # numpy array
            ),
        ]

        # Mock store.nodes.get_nodes to return our nodes
        mock_store.nodes = Mock()
        mock_store.nodes.get_nodes.return_value = nodes

        # Create query embedding
        query_embedding = [0.15, 0.25, 0.35]

        # Compute scores - should handle both types
        scores: dict[str, float] = {}
        node_ids = {"node1", "node2"}

        service._compute_remaining_scores(
            query_embedding=query_embedding,
            node_ids=node_ids,
            scores=scores,
        )

        # Both nodes should have scores
        assert "node1" in scores
        assert "node2" in scores
        assert scores["node1"] != 0.0
        assert scores["node2"] != 0.0
