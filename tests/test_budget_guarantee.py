"""Test budget guarantees in retrieval and assembly."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import Retriever
from ragzoom.store import Store


class TestBudgetGuarantee:
    """Test that budget guarantees are enforced by construction."""

    @pytest.fixture
    def setup_system(self):
        """Set up a test system with mocked API."""
        # Mock OpenAI clients
        with (
            patch("ragzoom.index.AsyncOpenAI") as mock_index_client,
            patch("ragzoom.retrieve.OpenAI") as mock_retrieve_client,
            patch("ragzoom.assemble.OpenAI") as mock_assemble_client,
            patch("chromadb.PersistentClient"),
        ):

            # Setup async mocks for indexing
            async def mock_embeddings_create(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 384) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 384)])

            async def mock_chat_create(*args, **kwargs):
                # Generate predictable summaries with <<<MID>>> delimiter
                messages = kwargs.get("messages", [])
                content = messages[-1]["content"] if messages else ""

                # Extract token count hint from the prompt
                if "approximately 50 tokens" in content:
                    return Mock(
                        choices=[
                            Mock(
                                message=Mock(
                                    content="Short left summary. <<<MID>>> Short right summary."
                                )
                            )
                        ]
                    )
                else:
                    return Mock(
                        choices=[
                            Mock(
                                message=Mock(
                                    content="This is the left half summary text. <<<MID>>> This is the right half summary text."
                                )
                            )
                        ]
                    )

            # Setup sync mocks for retrieval/assembly
            def mock_embeddings_create_sync(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 384) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 384)])

            # Configure mocks
            mock_embeddings_async = Mock()
            mock_embeddings_async.create = Mock(side_effect=mock_embeddings_create)

            mock_embeddings_sync = Mock()
            mock_embeddings_sync.create = Mock(side_effect=mock_embeddings_create_sync)

            mock_chat_async = Mock()
            mock_chat_async.completions = Mock()
            mock_chat_async.completions.create = Mock(side_effect=mock_chat_create)

            # Set up clients
            instance_async = Mock()
            instance_async.embeddings = mock_embeddings_async
            instance_async.chat = mock_chat_async
            mock_index_client.return_value = instance_async

            for mock_client in [mock_retrieve_client, mock_assemble_client]:
                instance_sync = Mock()
                instance_sync.embeddings = mock_embeddings_sync
                mock_client.return_value = instance_sync

            # Create test config with specific budget and temporary directory for ChromaDB
            with tempfile.TemporaryDirectory() as temp_dir:
                config = RagZoomConfig(
                    openai_api_key="test-key",
                    sqlite_database_url="sqlite:///:memory:",
                    chroma_persist_directory=temp_dir,
                    leaf_tokens=200,  # Standard leaf size
                    budget_tokens=1000,  # Strict budget for testing
                    adjacent_context_tokens=50,
                )

                store = Store(config)
                tree_builder = TreeBuilder(config, store)
                retriever = Retriever(config, store, tree_builder)
                assembler = Assembler(config, store)

                yield config, store, tree_builder, retriever, assembler

                # Close store to prevent file handle leaks
                store.close()

    def test_budget_never_exceeded_worst_case(self, setup_system):
        """Test that assembly never exceeds budget even in worst case."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a document that will build a multi-level tree
        # Each chunk is ~200 tokens, create enough for a 3-level tree
        chunk_text = "This is test content. " * 40  # ~200 tokens
        document = " ".join([chunk_text for _ in range(8)])  # 8 chunks = 3 levels

        tree_builder.add_document(document, "test-doc")

        # Test multiple queries with budget-only mode
        test_queries = [
            "test content",
            "this is",
            "random query that might not match well",
            "test",
        ]

        for query in test_queries:
            # Retrieve with budget constraint
            result = retriever.retrieve(query, budget_tokens=config.budget_tokens)

            # Assemble the result
            assembled_text = assembler.assemble(result)
            token_count = assembler.get_token_count(assembled_text)

            # CRITICAL: Token count must NEVER exceed budget
            assert (
                token_count <= config.budget_tokens
            ), f"Budget exceeded for query '{query}': {token_count} > {config.budget_tokens}"

            # Also verify by encoding directly
            actual_tokens = assembler.tokenizer.encode(assembled_text)
            assert (
                len(actual_tokens) <= config.budget_tokens
            ), f"Actual token count exceeds budget: {len(actual_tokens)} > {config.budget_tokens}"

    def test_worst_case_parent_child_extraction(self, setup_system):
        """Test worst case where parent-child extraction could double content."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a simple tree with known structure
        leaf1_text = "First leaf content. " * 40  # ~200 tokens
        leaf2_text = "Second leaf content. " * 40  # ~200 tokens

        # Create parent first with <<<MID>>> delimiter
        parent_text = "Summary of first leaf. <<<MID>>> Summary of second leaf."
        store.add_node(
            node_id="1_0_400_parent",
            text=parent_text,
            span_start=0,
            span_end=400,
            parent_id=None,
            document_id="test-doc",
            embedding=[0.15] * 384,
            mid_offset=len("Summary of first leaf. "),
        )

        # Then create children pointing to parent
        store.add_node(
            node_id="0_0_200_leaf1",
            text=leaf1_text,
            span_start=0,
            span_end=200,
            parent_id="1_0_400_parent",
            document_id="test-doc",
            embedding=[0.1] * 384,
        )

        store.add_node(
            node_id="0_200_400_leaf2",
            text=leaf2_text,
            span_start=200,
            span_end=400,
            parent_id="1_0_400_parent",
            document_id="test-doc",
            embedding=[0.2] * 384,
        )

        # Update parent to reference children
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            parent_node = session.query(TreeNode).filter_by(id="1_0_400_parent").first()
            if parent_node:
                parent_node.left_child_id = "0_0_200_leaf1"
                parent_node.right_child_id = "0_200_400_leaf2"
                session.commit()

        # Update children to point to parent
        # Note: Store doesn't have update_node_parent, nodes already have parent_id set

        # Worst case: retriever selects parent + one child
        # This tests the <<<MID>>> extraction logic
        result = retriever.retrieve("first leaf", n_max=2, budget_tokens=500)

        # Force a worst-case frontier (parent + left child)
        result.frontier_nodes = ["1_0_400_parent", "0_0_200_leaf1"]

        # Assemble and check budget
        assembled_text = assembler.assemble(result)
        token_count = assembler.get_token_count(assembled_text)

        assert token_count <= 500, f"Budget exceeded in worst case: {token_count} > 500"

        # Verify no content duplication
        assert (
            assembled_text.count("First leaf content.") <= 40
        ), "Content was duplicated"

    def test_conservative_n_max_calculation(self, setup_system):
        """Test that the conservative n_max calculation is reasonable and respects budget."""
        config, store, tree_builder, retriever, assembler = setup_system
        config.leaf_tokens = 100  # Use a predictable size for testing

        document = "This is test content for budget calculation. " * 500
        tree_builder.add_document(document, "doc-budget-test")

        # Test various budget sizes
        test_budgets = [500, 1000, 2000, 5000]

        for budget in test_budgets:
            # Calculate conservative n_max using the retriever's method
            conservative_n_max = retriever._calculate_conservative_n_max(
                budget, "doc-budget-test"
            )

            # Verify the calculation is at least 1
            assert (
                conservative_n_max >= 1
            ), f"n_max calculation should be at least 1 for budget {budget}"

            # Now, verify the outcome: does using this n_max actually respect the budget?
            result = retriever.retrieve(
                "test",
                n_max=conservative_n_max,
                budget_tokens=budget,
                document_id="doc-budget-test",
            )

            # The retriever's own internal enforcement should already have trimmed the frontier
            # Let's assemble and get the final count
            assembled_text = assembler.assemble(result)
            final_token_count = assembler.get_token_count(assembled_text)

            assert (
                final_token_count <= budget
            ), f"Budget {budget} exceeded with conservative_n_max={conservative_n_max}, final tokens={final_token_count}"

    def test_mixed_mode_budget_plus_n_max(self, setup_system):
        """Test mixed mode where both budget and n_max are specified."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a document
        document = "Test content. " * 200
        tree_builder.add_document(document, "test-doc")

        # Specify both budget and n_max
        budget = 800
        n_max = 10  # Potentially too many nodes for budget

        # Retrieve with both constraints
        result = retriever.retrieve("test", n_max=n_max, budget_tokens=budget)

        # Should respect budget with DP algorithm
        assert result.frontier_segments is not None

        # Assemble and verify budget
        assembled_text = assembler.assemble(result)
        token_count = assembler.get_token_count(assembled_text)
        assert (
            token_count <= budget
        ), f"Budget exceeded with mixed mode: {token_count} > {budget}"

    def test_n_max_only_mode(self, setup_system):
        """Test n_max only mode (no budget enforcement)."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a document
        document = "Test content. " * 200
        tree_builder.add_document(document, "test-doc")

        # Retrieve with only n_max (no budget)
        n_max = 5
        result = retriever.retrieve("test", n_max=n_max, budget_tokens=None)

        # Should have segments from DP algorithm
        assert result.frontier_segments is not None

        # Assembly should work without budget constraints
        assembled_text = assembler.assemble(result)
        assert len(assembled_text) > 0
