"""Test to verify cross-document contamination bug (issue #222) is fixed."""

import pytest

from ragzoom.assemble import Assembler
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from tests.utils import mock_openai_context


class TestContaminationBug:
    """Comprehensive test for the cross-document contamination bug fix."""

    @pytest.fixture
    def mock_openai(self):
        """Mock OpenAI API with document-specific embeddings."""
        # Different embeddings for different documents
        embedding_rules = {
            "Ahab": [0.9, 0.1] + [0.0] * 1534,  # Strong signal for Moby Dick
            "whale": [0.8, 0.2] + [0.0] * 1534,  # Related to Moby Dick
            "Ishmael": [0.85, 0.15] + [0.0] * 1534,  # Related to Moby Dick
            "Bilbo": [0.1, 0.9] + [0.0] * 1534,  # Strong signal for Hobbit
            "hobbit": [0.2, 0.8] + [0.0] * 1534,  # Related to Hobbit
            "Gandalf": [0.15, 0.85] + [0.0] * 1534,  # Related to Hobbit
        }
        with mock_openai_context(embedding_rules) as mocks:
            yield mocks

    @pytest.fixture
    def setup(self, mock_openai, store, base_config):
        """Create test environment with multiple documents."""
        from openai import OpenAI

        from ragzoom.retrieval.budget_planner import BudgetPlanner
        from ragzoom.retrieval.embedding_service import EmbeddingService

        tree_builder = TreeBuilder(
            base_config.index_config, store, api_key=base_config.openai_api_key
        )

        # Create services for Retriever
        client = OpenAI(api_key=base_config.openai_api_key.get_secret_value())
        embedding_service = EmbeddingService(
            client, store, base_config.query_config.embedding_model
        )
        budget_planner = BudgetPlanner(
            store, base_config.index_config.target_chunk_tokens
        )

        yield base_config, store, tree_builder, embedding_service, budget_planner

    def test_no_cross_document_contamination(self, setup):
        """Test that queries return content ONLY from the specified document."""
        config, store, tree_builder, embedding_service, budget_planner = setup

        # Index Moby Dick content
        moby_dick_text = """
        Call me Ishmael. Captain Ahab was obsessed with the white whale.
        The whale was massive and powerful. Ahab's quest consumed him.
        The crew feared both Ahab and the whale equally.
        """

        # Index Hobbit content
        hobbit_text = """
        In a hole in the ground there lived a hobbit. Bilbo Baggins was
        his name. Gandalf the wizard visited him one morning. Together
        they would embark on an unexpected journey.
        """

        # Index both documents
        tree_builder.add_document(moby_dick_text, document_id="moby_dick.txt")
        tree_builder.add_document(hobbit_text, document_id="the_hobbit.txt")

        # CRITICAL TEST 1: Query Moby Dick document about Ahab
        moby_store = store.for_document("moby_dick.txt")
        moby_retriever = Retriever(
            config.query_config,
            moby_store,
            embedding_service,
            budget_planner,
        )
        moby_assembler = Assembler(moby_store)

        # Query specifically about Ahab (Moby Dick character)
        moby_result = moby_retriever.retrieve("Tell me about Captain Ahab")
        moby_summary = moby_assembler.assemble(moby_result)

        # Verify NO Hobbit content appears
        assert "Bilbo" not in moby_summary, "Found Hobbit content in Moby Dick query!"
        assert "Gandalf" not in moby_summary, "Found Hobbit content in Moby Dick query!"
        assert (
            "hobbit" not in moby_summary.lower()
        ), "Found Hobbit content in Moby Dick query!"

        # Verify Moby Dick content IS present
        assert (
            "Ahab" in moby_summary
            or "whale" in moby_summary
            or "Ishmael" in moby_summary
        ), "Missing Moby Dick content in Moby Dick query!"

        # CRITICAL TEST 2: Query Hobbit document about Bilbo
        hobbit_store = store.for_document("the_hobbit.txt")
        hobbit_retriever = Retriever(
            config.query_config,
            hobbit_store,
            embedding_service,
            budget_planner,
        )
        hobbit_assembler = Assembler(hobbit_store)

        # Query specifically about Bilbo (Hobbit character)
        hobbit_result = hobbit_retriever.retrieve("Tell me about Bilbo Baggins")
        hobbit_summary = hobbit_assembler.assemble(hobbit_result)

        # Verify NO Moby Dick content appears
        assert "Ahab" not in hobbit_summary, "Found Moby Dick content in Hobbit query!"
        assert (
            "whale" not in hobbit_summary.lower()
        ), "Found Moby Dick content in Hobbit query!"
        assert (
            "Ishmael" not in hobbit_summary
        ), "Found Moby Dick content in Hobbit query!"

        # Verify Hobbit content IS present
        assert (
            "Bilbo" in hobbit_summary
            or "Gandalf" in hobbit_summary
            or "hobbit" in hobbit_summary.lower()
        ), "Missing Hobbit content in Hobbit query!"

        # CRITICAL TEST 3: Verify all nodes in results belong to correct document
        for node_id in moby_result.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "moby_dick.txt"
            ), f"Node {node_id} from wrong document in Moby Dick results!"

        for node_id in hobbit_result.node_ids:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "the_hobbit.txt"
            ), f"Node {node_id} from wrong document in Hobbit results!"

    def test_coverage_map_isolation(self, setup):
        """Test that coverage maps don't include nodes from other documents."""
        config, store, tree_builder, embedding_service, budget_planner = setup

        # Index two documents
        doc1_text = "Document one contains unique content about alpha topics."
        doc2_text = "Document two contains unique content about beta topics."

        tree_builder.add_document(doc1_text, document_id="doc1.txt")
        tree_builder.add_document(doc2_text, document_id="doc2.txt")

        # Create retriever for doc1
        doc1_store = store.for_document("doc1.txt")
        retriever1 = Retriever(
            config.query_config,
            doc1_store,
            embedding_service,
            budget_planner,
        )

        # Retrieve from doc1
        result1 = retriever1.retrieve("alpha topics")

        # Verify coverage map only contains doc1 nodes
        for node_id in result1.coverage_map:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "doc1.txt"
            ), f"Coverage map contains node from wrong document: {node.document_id}"

        # Create retriever for doc2
        doc2_store = store.for_document("doc2.txt")
        retriever2 = Retriever(
            config.query_config,
            doc2_store,
            embedding_service,
            budget_planner,
        )

        # Retrieve from doc2
        result2 = retriever2.retrieve("beta topics")

        # Verify coverage map only contains doc2 nodes
        for node_id in result2.coverage_map:
            node = store.nodes.get_node(node_id)
            assert (
                node.document_id == "doc2.txt"
            ), f"Coverage map contains node from wrong document: {node.document_id}"

        # Verify no overlap between coverage maps
        overlap = set(result1.coverage_map.keys()) & set(result2.coverage_map.keys())
        assert len(overlap) == 0, f"Coverage maps have overlapping nodes: {overlap}"
