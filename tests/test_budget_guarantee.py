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
            assembled_text, token_count = assembler.assemble_with_budget(result)

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
            depth=1,
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
            depth=0,
            span_start=0,
            span_end=200,
            parent_id="1_0_400_parent",
            document_id="test-doc",
            embedding=[0.1] * 384,
        )

        store.add_node(
            node_id="0_200_400_leaf2",
            text=leaf2_text,
            depth=0,
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
        assembled_text, token_count = assembler.assemble_with_budget(
            result, token_budget=500
        )

        assert token_count <= 500, f"Budget exceeded in worst case: {token_count} > 500"

        # Verify no content duplication
        assert (
            assembled_text.count("First leaf content.") <= 40
        ), "Content was duplicated"

    def test_conservative_n_max_calculation(self, setup_system):
        """Test that conservative n_max calculation prevents overruns."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Test various budget sizes
        test_budgets = [500, 1000, 2000, 5000]

        for budget in test_budgets:
            # Calculate conservative n_max using the retriever's method
            conservative_n_max = retriever._calculate_conservative_n_max(budget)

            # Verify the calculation makes sense
            assert (
                conservative_n_max >= 1
            ), f"n_max calculation failed for budget {budget}"

            # Calculate expected safe factor
            if config.slope_cap:
                safe_factor = config.slope_cap_size + 2  # ±1→3, ±2→4, etc
            else:
                safe_factor = 8  # Fallback for disabled slope cap

            # For very small budgets, even 1 node might exceed budget
            if budget < config.leaf_tokens * safe_factor:
                # In this case, we still return 1 but assembly will need to handle it
                assert (
                    conservative_n_max == 1
                ), f"Should return 1 for small budget {budget}"
            else:
                # Verify this n_max would never exceed budget with safe factor
                worst_case_tokens = (
                    conservative_n_max * config.leaf_tokens * safe_factor
                )
                assert (
                    worst_case_tokens <= budget
                ), f"Conservative n_max would exceed budget: {worst_case_tokens} > {budget}"

                # Also verify the exact calculation
                expected_n_max = max(1, budget // (config.leaf_tokens * safe_factor))
                assert (
                    conservative_n_max == expected_n_max
                ), f"n_max calculation incorrect: {conservative_n_max} != {expected_n_max}"

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

        # Should respect n_max but drop nodes if needed for budget
        assert len(result.frontier_nodes) <= n_max

        # Assemble and verify budget
        assembled_text, token_count = assembler.assemble_with_budget(
            result, token_budget=budget
        )
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

        # Should respect n_max
        assert len(result.frontier_nodes) <= n_max

        # Assembly should work without budget constraints
        assembled_text = assembler.assemble(result)
        assert len(assembled_text) > 0

    def test_empty_frontier_edge_case(self, setup_system):
        """Test edge case with empty frontier."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Don't add any documents
        result = retriever.retrieve("test query")

        # Should handle gracefully
        assembled_text, token_count = assembler.assemble_with_budget(result)
        assert token_count == 0
        assert assembled_text == ""

    def test_worst_case_parent_grandchild_3x_bound(self, setup_system):
        """Test the worst-case scenario that requires 3x bound: parent + grandchild with slope-cap ±1."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a tree structure demonstrating the worst case
        # Tree structure:
        #   Depth 2: Root (grandparent)
        #   Depth 1: Parent nodes
        #   Depth 0: Leaf nodes (grandchildren)

        # Create leaves first (200 tokens each)
        leaf1_text = "Leaf one content. " * 40  # ~200 tokens
        leaf2_text = "Leaf two content. " * 40  # ~200 tokens
        leaf3_text = "Leaf three content. " * 40  # ~200 tokens
        leaf4_text = "Leaf four content. " * 40  # ~200 tokens

        # Add leaves
        store.add_node(
            "0_0_200_leaf1",
            leaf1_text,
            [0.1] * 384,
            0,
            0,
            200,
            None,
            None,
            None,
            "test-doc",
        )
        store.add_node(
            "0_200_400_leaf2",
            leaf2_text,
            [0.2] * 384,
            0,
            200,
            400,
            None,
            None,
            None,
            "test-doc",
        )
        store.add_node(
            "0_400_600_leaf3",
            leaf3_text,
            [0.3] * 384,
            0,
            400,
            600,
            None,
            None,
            None,
            "test-doc",
        )
        store.add_node(
            "0_600_800_leaf4",
            leaf4_text,
            [0.4] * 384,
            0,
            600,
            800,
            None,
            None,
            None,
            "test-doc",
        )

        # Create parent nodes with <<<MID>>> delimiter
        parent1_text = "Summary of leaf one. <<<MID>>> Summary of leaf two."
        parent2_text = "Summary of leaf three. <<<MID>>> Summary of leaf four."

        store.add_node(
            "1_0_400_parent1",
            parent1_text,
            [0.15] * 384,
            1,
            0,
            400,
            None,
            None,
            None,
            "test-doc",
            mid_offset=len("Summary of leaf one. "),
        )
        store.add_node(
            "1_400_800_parent2",
            parent2_text,
            [0.35] * 384,
            1,
            400,
            800,
            None,
            None,
            None,
            "test-doc",
            mid_offset=len("Summary of leaf three. "),
        )

        # Create grandparent
        grandparent_text = "Overall summary left. <<<MID>>> Overall summary right."
        store.add_node(
            "2_0_800_root",
            grandparent_text,
            [0.25] * 384,
            2,
            0,
            800,
            None,
            None,
            None,
            "test-doc",
            mid_offset=len("Overall summary left. "),
        )

        # Update parent-child relationships
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            # Update leaves to point to parents
            for leaf_id, parent_id in [
                ("0_0_200_leaf1", "1_0_400_parent1"),
                ("0_200_400_leaf2", "1_0_400_parent1"),
                ("0_400_600_leaf3", "1_400_800_parent2"),
                ("0_600_800_leaf4", "1_400_800_parent2"),
            ]:
                leaf = session.query(TreeNode).filter_by(id=leaf_id).first()
                if leaf:
                    leaf.parent_id = parent_id

            # Update parents to have children and point to grandparent
            parent1 = session.query(TreeNode).filter_by(id="1_0_400_parent1").first()
            if parent1:
                parent1.left_child_id = "0_0_200_leaf1"
                parent1.right_child_id = "0_200_400_leaf2"
                parent1.parent_id = "2_0_800_root"

            parent2 = session.query(TreeNode).filter_by(id="1_400_800_parent2").first()
            if parent2:
                parent2.left_child_id = "0_400_600_leaf3"
                parent2.right_child_id = "0_600_800_leaf4"
                parent2.parent_id = "2_0_800_root"

            # Update grandparent to have children
            root = session.query(TreeNode).filter_by(id="2_0_800_root").first()
            if root:
                root.left_child_id = "1_0_400_parent1"
                root.right_child_id = "1_400_800_parent2"

            session.commit()

        # Test the worst case: frontier with parent + grandchild (slope-cap ±1)
        # This simulates when retrieval selects nodes at depths 1 and 0
        result = retriever.retrieve("leaf one parent", n_max=10, budget_tokens=700)

        # Force a worst-case frontier: parent at depth 1 + grandchild at depth 0
        result.frontier_nodes = [
            "1_0_400_parent1",
            "0_400_600_leaf3",
        ]  # Parent + non-child leaf

        # With parent outputting its left child (200 tokens) + grandchild (200 tokens) = 400 tokens
        # But parent could output BOTH children in worst case = 600 tokens for just the parent!
        assembled_text, token_count = assembler.assemble_with_budget(
            result, token_budget=700
        )

        # Should not exceed budget
        assert token_count <= 700, f"Budget exceeded in worst case: {token_count} > 700"

        # Verify the 3x bound is necessary
        # Parent node could expand to ~600 tokens (both children), grandchild to 200 tokens
        # So 2 nodes could require 800 tokens, meaning we need 3x bound per "average" node

    def test_drop_strategy_vs_truncate(self, setup_system):
        """Test that drop strategy preserves coherence better than truncation."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a document that would exceed budget
        chunk_text = "This is test content. " * 40  # ~200 tokens per chunk
        document = " ".join([chunk_text for _ in range(6)])  # 6 chunks

        tree_builder.add_document(document, "test-doc")

        # Test with very small budget to force strategy usage
        small_budget = 400  # Force dropping/truncation

        # Test drop strategy
        config.budget_strategy = "drop"
        result_drop = retriever.retrieve("test content", budget_tokens=small_budget)
        summary_drop, tokens_drop = assembler.assemble_with_budget(
            result_drop, token_budget=small_budget
        )

        # Verify drop strategy stays within budget
        assert (
            tokens_drop <= small_budget
        ), f"Drop strategy exceeded budget: {tokens_drop} > {small_budget}"

        # Test truncate strategy
        config.budget_strategy = "truncate"
        result_truncate = retriever.retrieve("test content", budget_tokens=small_budget)
        summary_truncate, tokens_truncate = assembler.assemble_with_budget(
            result_truncate, token_budget=small_budget
        )

        # Verify truncate strategy stays within budget
        assert (
            tokens_truncate <= small_budget
        ), f"Truncate strategy exceeded budget: {tokens_truncate} > {small_budget}"

        # Drop strategy should produce complete sentences while truncate might cut mid-sentence
        # This is hard to test directly, but we can verify both work
        assert len(summary_drop) > 0, "Drop strategy produced empty summary"
        assert len(summary_truncate) > 0, "Truncate strategy produced empty summary"

    def test_drop_strategy_maintains_slope_cap(self, setup_system):
        """Test that drop strategy maintains slope cap after trimming nodes."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a scenario that could violate slope cap after dropping bridge nodes
        # Create nodes at depths 0, 1, 2 to test the "bridge node" scenario

        # Add nodes manually to create the problematic pattern
        leaf1_text = "Leaf content at depth 0. " * 40  # ~200 tokens
        bridge_text = "Bridge summary at depth 1. " * 20  # ~100 tokens
        deep_text = "Deep summary at depth 2. " * 60  # ~300 tokens

        store.add_node(
            "0_0_200_leaf",
            leaf1_text,
            [0.1] * 384,
            0,
            0,
            200,
            None,
            None,
            None,
            "test-doc",
        )
        store.add_node(
            "1_200_400_bridge",
            bridge_text,
            [0.2] * 384,
            1,
            200,
            400,
            None,
            None,
            None,
            "test-doc",
        )
        store.add_node(
            "2_400_600_deep",
            deep_text,
            [0.3] * 384,
            2,
            400,
            600,
            None,
            None,
            None,
            "test-doc",
        )

        # Create a retrieval result with this frontier pattern
        from ragzoom.retrieve import RetrievalResult

        result = RetrievalResult(
            node_ids=["0_0_200_leaf", "1_200_400_bridge", "2_400_600_deep"],
            scores={
                "0_0_200_leaf": 0.9,
                "1_200_400_bridge": 0.3,
                "2_400_600_deep": 0.8,
            },  # Bridge has low score
            coverage_map={
                "0_0_200_leaf": True,
                "1_200_400_bridge": True,
                "2_400_600_deep": True,
            },
            frontier_nodes=["0_0_200_leaf", "1_200_400_bridge", "2_400_600_deep"],
        )

        # Set budget low enough to force dropping the bridge node (lowest utility)
        config.budget_strategy = "drop"
        small_budget = (
            450  # Should keep leaf (200) + deep (300) = 500, but drop bridge (100)
        )

        # Assemble with drop strategy
        assembled_text, token_count = assembler.assemble_with_budget(
            result, token_budget=small_budget
        )

        # Should not exceed budget
        assert (
            token_count <= small_budget
        ), f"Budget exceeded: {token_count} > {small_budget}"

        # Should produce valid output despite slope cap challenge
        assert (
            len(assembled_text) > 0
        ), "Drop strategy with slope cap produced empty summary"

    def test_post_slope_cap_budget_overflow_prevention(self, setup_system):
        """Test that budget is not exceeded after slope cap adds ancestor nodes."""
        config, store, tree_builder, retriever, assembler = setup_system

        # Create a scenario where slope cap will add ancestor nodes after budget trim
        # Tree: Root -> Parent -> Child (depths 2, 1, 0)
        # If we select root + child initially, slope cap will add parent to fix ±2 violation

        child_text = "Child content. " * 50  # ~200 tokens
        parent_text = "Parent summary. " * 50  # ~200 tokens
        root_text = "Root summary. " * 100  # ~400 tokens

        # Add nodes manually
        store.add_node(
            "0_0_200_child",
            child_text,
            [0.1] * 384,
            0,
            0,
            200,
            None,
            None,
            None,
            "test-doc",
        )
        store.add_node(
            "1_0_200_parent",
            parent_text,
            [0.2] * 384,
            1,
            0,
            200,
            None,
            None,
            None,
            "test-doc",
        )
        store.add_node(
            "2_0_200_root",
            root_text,
            [0.3] * 384,
            2,
            0,
            200,
            None,
            None,
            None,
            "test-doc",
        )

        # Set up parent-child relationships
        with store.SessionLocal() as session:
            from ragzoom.store import TreeNode

            child = session.query(TreeNode).filter_by(id="0_0_200_child").first()
            if child:
                child.parent_id = "1_0_200_parent"

            parent = session.query(TreeNode).filter_by(id="1_0_200_parent").first()
            if parent:
                parent.left_child_id = "0_0_200_child"
                parent.parent_id = "2_0_200_root"

            root = session.query(TreeNode).filter_by(id="2_0_200_root").first()
            if root:
                root.left_child_id = "1_0_200_parent"

            session.commit()

        # Create retrieval result with frontier that violates slope cap (root + child = ±2)
        from ragzoom.retrieve import RetrievalResult

        result = RetrievalResult(
            node_ids=["2_0_200_root", "0_0_200_child"],
            scores={"2_0_200_root": 0.9, "0_0_200_child": 0.8, "1_0_200_parent": 0.7},
            coverage_map={"2_0_200_root": True, "0_0_200_child": True},
            frontier_nodes=[
                "2_0_200_root",
                "0_0_200_child",
            ],  # This violates slope cap ±2
        )

        # Set very tight budget that root + child would fit, but adding parent would exceed
        config.budget_strategy = "drop"
        tight_budget = 550  # Root(400) + Child(200) = 600, but Parent adds 200 more

        # Assemble - slope cap should add parent node, potentially exceeding budget
        assembled_text, token_count = assembler.assemble_with_budget(
            result, token_budget=tight_budget
        )

        # CRITICAL: Must never exceed budget, even after slope cap modifications
        assert (
            token_count <= tight_budget
        ), f"Budget exceeded after slope cap: {token_count} > {tight_budget}"

        # Should still produce valid output
        assert (
            len(assembled_text) > 0
        ), "Post-slope-cap budget fix produced empty summary"
