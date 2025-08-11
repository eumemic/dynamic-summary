"""Test budget guarantees in retrieval and assembly."""

import tempfile
from unittest.mock import Mock, patch

import pytest

from ragzoom.assemble import Assembler
from ragzoom.config import IndexConfig, OperationalConfig, QueryConfig
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
            patch("chromadb.PersistentClient"),
        ):

            # Setup async mocks for indexing
            async def mock_embeddings_create(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 1536)])

            async def mock_chat_create(*args, **kwargs):
                # Generate predictable summaries
                messages = kwargs.get("messages", [])
                content = messages[-1]["content"] if messages else ""

                # Extract token count hint from the prompt
                if "approximately 50 tokens" in content:
                    return Mock(
                        choices=[
                            Mock(
                                message=Mock(content="Short summary of both children.")
                            )
                        ]
                    )
                else:
                    return Mock(
                        choices=[
                            Mock(
                                message=Mock(
                                    content="This is the combined summary text for both children."
                                )
                            )
                        ]
                    )

            # Setup sync mocks for retrieval/assembly
            def mock_embeddings_create_sync(*args, **kwargs):
                input_data = kwargs.get("input", args[0] if args else "")
                if isinstance(input_data, list):
                    return Mock(data=[Mock(embedding=[0.1] * 1536) for _ in input_data])
                else:
                    return Mock(data=[Mock(embedding=[0.1] * 1536)])

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

            # Set up sync client for retrieve
            instance_sync = Mock()
            instance_sync.embeddings = mock_embeddings_sync
            mock_retrieve_client.return_value = instance_sync

            # Create test configs with specific budget and temporary directory for ChromaDB
            with tempfile.TemporaryDirectory() as temp_dir:
                index_config = IndexConfig(
                    target_chunk_tokens=200,  # Standard leaf size
                    prev_context_tokens=50,
                )
                query_config = QueryConfig(
                    budget_tokens=1000,  # Strict budget for testing
                )
                operational_config = OperationalConfig(
                    openai_api_key="test-key",
                    sqlite_database_url="sqlite:///:memory:",
                    chroma_persist_directory=temp_dir,
                )

                store = Store(
                    operational_config,
                    embedding_model=index_config.embedding_model,
                )
                tree_builder = TreeBuilder(
                    index_config,
                    store,
                    api_key=operational_config.openai_api_key,
                )
                retriever = Retriever(
                    query_config,
                    index_config,
                    store,
                    api_key=operational_config.openai_api_key,
                )
                assembler = Assembler(store)

                yield (
                    index_config,
                    query_config,
                    operational_config,
                ), store, tree_builder, retriever, assembler

                # Close store to prevent file handle leaks
                store.close()

    def test_budget_never_exceeded_worst_case(self, setup_system):
        """Test that assembly never exceeds budget even in worst case."""
        (
            (index_config, query_config, operational_config),
            store,
            tree_builder,
            retriever,
            assembler,
        ) = setup_system

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
            result = retriever.retrieve(query, budget_tokens=query_config.budget_tokens)

            # Assemble the result
            assembled_text = assembler.assemble(result)
            token_count = assembler.get_token_count(assembled_text)

            # CRITICAL: Token count must NEVER exceed budget
            assert (
                token_count <= query_config.budget_tokens
            ), f"Budget exceeded for query '{query}': {token_count} > {query_config.budget_tokens}"

            # Also verify by encoding directly
            actual_tokens = assembler.tokenizer.encode(assembled_text)
            assert (
                len(actual_tokens) <= query_config.budget_tokens
            ), f"Actual token count exceeds budget: {len(actual_tokens)} > {query_config.budget_tokens}"

    def test_worst_case_parent_child_extraction(self, setup_system):
        """Test worst case where parent-child extraction could double content."""
        (
            (index_config, query_config, operational_config),
            store,
            tree_builder,
            retriever,
            assembler,
        ) = setup_system

        # Create a simple tree with known structure
        leaf1_text = "First leaf content. " * 40  # ~200 tokens
        leaf2_text = "Second leaf content. " * 40  # ~200 tokens

        # Create parent node
        parent_text = "Summary of first and second leaves."
        store.add_node(
            node_id="1_0_400_parent",
            text=parent_text,
            span_start=0,
            span_end=400,
            parent_id=None,
            document_id="test-doc",
            embedding=[0.15] * 1536,
        )

        # Then create children pointing to parent
        store.add_node(
            node_id="0_0_200_leaf1",
            text=leaf1_text,
            span_start=0,
            span_end=200,
            parent_id="1_0_400_parent",
            document_id="test-doc",
            embedding=[0.1] * 1536,
        )

        store.add_node(
            node_id="0_200_400_leaf2",
            text=leaf2_text,
            span_start=200,
            span_end=400,
            parent_id="1_0_400_parent",
            document_id="test-doc",
            embedding=[0.2] * 1536,
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

        # Test that retriever respects budget even with parent + child nodes
        result = retriever.retrieve("first leaf", n_max=2, budget_tokens=500)

        # Note: Cannot force a specific tiling anymore since tiling field is computed by DP
        # This test may need to be redesigned to work with the new tiling-based approach

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
        (
            (index_config, query_config, operational_config),
            store,
            tree_builder,
            retriever,
            assembler,
        ) = setup_system

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

            # The retriever's own internal enforcement should already have trimmed the tiling
            # Let's assemble and get the final count
            assembled_text = assembler.assemble(result)
            final_token_count = assembler.get_token_count(assembled_text)

            assert (
                final_token_count <= budget
            ), f"Budget {budget} exceeded with conservative_n_max={conservative_n_max}, final tokens={final_token_count}"

    def test_mixed_mode_budget_plus_n_max(self, setup_system):
        """Test mixed mode where both budget and n_max are specified."""
        (
            (index_config, query_config, operational_config),
            store,
            tree_builder,
            retriever,
            assembler,
        ) = setup_system

        # Create a document
        document = "Test content. " * 200
        tree_builder.add_document(document, "test-doc")

        # Specify both budget and n_max
        budget = 800
        n_max = 10  # Potentially too many nodes for budget

        # Retrieve with both constraints
        result = retriever.retrieve("test", n_max=n_max, budget_tokens=budget)

        # Should respect budget with DP algorithm
        assert result.tiling is not None

        # Assemble and verify budget
        assembled_text = assembler.assemble(result)
        token_count = assembler.get_token_count(assembled_text)
        assert (
            token_count <= budget
        ), f"Budget exceeded with mixed mode: {token_count} > {budget}"

    def test_n_max_only_mode(self):
        """Test n_max only mode (no budget enforcement)."""
        from unittest.mock import Mock, patch

        from tests.mock_store import SimpleMockStore

        # Mock OpenAI clients
        with (patch("ragzoom.retrieve.OpenAI") as mock_retrieve_client,):

            # Setup sync mock for retrieval
            mock_embeddings = Mock()
            mock_embeddings.create = Mock(
                return_value=Mock(data=[Mock(embedding=[0.5] * 1536)])
            )

            instance_retrieve = Mock()
            instance_retrieve.embeddings = mock_embeddings
            mock_retrieve_client.return_value = instance_retrieve

            # No OpenAI setup needed for assembler

            index_config = IndexConfig(target_chunk_tokens=200)
            query_config = QueryConfig(budget_tokens=1000)
            operational_config = OperationalConfig(openai_api_key="test-key")
            store = SimpleMockStore(
                config=(index_config, query_config, operational_config)
            )
            retriever = Retriever(
                query_config,
                index_config,
                store,
                api_key=operational_config.openai_api_key,
            )
            assembler = Assembler(store)

            # Create a simple tree structure
            # Root
            store.add_node(
                node_id="root",
                text="Root summary of the document",
                span_start=0,
                span_end=2800,
                parent_id=None,
                document_id="test-doc",
                embedding=[0.1] * 1536,
                left_child_id="leaf1",
                right_child_id="leaf2",
            )

            # Leaf nodes
            store.add_node(
                node_id="leaf1",
                text="Test content. " * 100,  # ~200 tokens
                span_start=0,
                span_end=1400,
                parent_id="root",
                document_id="test-doc",
                embedding=[0.9] * 1536,  # High similarity to query
            )

            store.add_node(
                node_id="leaf2",
                text="Other content. " * 100,  # ~200 tokens
                span_start=1400,
                span_end=2800,
                parent_id="root",
                document_id="test-doc",
                embedding=[0.2] * 1536,  # Low similarity
            )

            # Set up mock scores to simulate search results
            store.set_mock_scores(
                {
                    "leaf1": 0.9,  # High score for "test" query
                    "leaf2": 0.2,
                    "root": 0.5,
                }
            )

            # Retrieve with only n_max (no budget)
            n_max = 5
            result = retriever.retrieve(
                "test", n_max=n_max, budget_tokens=None, document_id="test-doc"
            )

            # Should have nodes from DP algorithm
            assert result.tiling is not None
            assert len(result.tiling) > 0

            # Assembly should work without budget constraints
            assembled_text = assembler.assemble(result)
            assert len(assembled_text) > 0


class TestBudgetValidation:
    """Test that budget validation catches overflows."""

    def test_budget_validation_catches_overflow(self):
        """Test that validation fails when tiling exceeds budget."""
        from ragzoom.validate import validate_tiling
        from tests.mock_store import SimpleMockStore

        index_config = IndexConfig(target_chunk_tokens=100)
        query_config = QueryConfig()
        operational_config = OperationalConfig()
        store = SimpleMockStore(config=(index_config, query_config, operational_config))

        # Create some nodes with known token costs
        # "test " * 20 = ~20 tokens
        store.add_node(
            node_id="node1",
            text="test " * 20,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            document_id="test-doc",
        )

        store.add_node(
            node_id="node2",
            text="test " * 30,
            embedding=[0.2] * 1536,
            span_start=100,
            span_end=200,
            document_id="test-doc",
        )

        # Create tiling that would exceed a small budget
        tiling = ["node1", "node2"]  # ~20 + ~30 = ~50 tokens

        # Validate with budget that's too small
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=40)

        assert error is not None
        assert "exceeds budget" in error
        assert "> 40 budget" in error

    def test_budget_validation_passes_within_budget(self):
        """Test that validation passes when tiling is within budget."""
        from ragzoom.validate import validate_tiling
        from tests.mock_store import SimpleMockStore

        index_config = IndexConfig(target_chunk_tokens=100)
        query_config = QueryConfig()
        operational_config = OperationalConfig()
        store = SimpleMockStore(config=(index_config, query_config, operational_config))

        # Create a node
        store.add_node(
            node_id="node1",
            text="test " * 10,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=50,
            document_id="test-doc",
        )

        # Create tiling within budget
        tiling = ["node1"]  # ~10 tokens

        # Validate with sufficient budget
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=100)

        assert error is None

    def test_budget_validation_with_parent_child(self):
        """Test budget validation with parent and child nodes."""
        from ragzoom.validate import validate_tiling
        from tests.mock_store import SimpleMockStore

        index_config = IndexConfig(target_chunk_tokens=100)
        query_config = QueryConfig()
        operational_config = OperationalConfig()
        store = SimpleMockStore(config=(index_config, query_config, operational_config))

        # Create child nodes
        store.add_node(
            node_id="left_child",
            text="left part " * 10,
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=100,
            document_id="test-doc",
        )

        store.add_node(
            node_id="right_child",
            text="right part " * 10,
            embedding=[0.1] * 1536,
            span_start=100,
            span_end=200,
            document_id="test-doc",
        )

        # Create parent node
        store.add_node(
            node_id="parent",
            text="Summary of left and right parts",
            embedding=[0.1] * 1536,
            span_start=0,
            span_end=200,
            document_id="test-doc",
            summary="Summary of left and right parts",
            left_child_id="left_child",
            right_child_id="right_child",
        )

        # Create tiling with child nodes
        tiling = ["left_child", "right_child"]  # ~20 tokens total

        # Should pass with budget of 50
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=50)
        assert error is None

        # Should fail with budget of 15
        error = validate_tiling(tiling, store, "test-doc", budget_tokens=15)
        assert error is not None
        assert "exceeds budget" in error
