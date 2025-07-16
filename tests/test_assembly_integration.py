"""Integration tests for the assembly pipeline to prevent repetitions and ensure correctness."""

from unittest.mock import MagicMock, patch

import pytest

from ragzoom.assemble import Assembler
from ragzoom.index import TreeBuilder
from ragzoom.retrieve import RetrievalResult, Retriever


class TestAssemblyIntegration:
    """Test the complete assembly pipeline."""

    @pytest.fixture
    def setup_components(self, base_config, store, monkeypatch):
        """Set up test components."""
        # Use the store fixture which automatically selects mock/real based on markers
        monkeypatch.setenv("RAGZOOM_OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("RAGZOOM_SLOPE_CAP", "true")

        # Mock OpenAI client
        mock_client = MagicMock()
        mock_async_client = MagicMock()

        yield base_config, store, mock_client, mock_async_client

    def test_no_duplicate_content_in_assembly(self, setup_components):
        """Test that assembled output contains no repeated content."""
        config, store, mock_client, _ = setup_components

        # Create a simple tree structure
        # Root
        #  / \
        # L   R

        # Add leaf nodes
        store.add_node(
            node_id="left_leaf",
            text="Sarah discovered a book in the attic.",
            embedding=[0.1] * 1536,
            depth=0,
            span_start=0,
            span_end=38,
            document_id="doc1",
        )

        store.add_node(
            node_id="right_leaf",
            text="She went to the library and met Mr. Chen.",
            embedding=[0.2] * 1536,
            depth=0,
            span_start=38,
            span_end=79,
            document_id="doc1",
        )

        # Add parent with mid delimiter
        parent_text = "Sarah discovered a book in the attic. <<<MID>>> She went to the library and met Mr. Chen."
        mid_offset = parent_text.find("<<<MID>>>")

        store.add_node(
            node_id="root",
            text=parent_text,
            embedding=[0.15] * 1536,
            depth=1,
            span_start=0,
            span_end=79,
            left_child_id="left_leaf",
            right_child_id="right_leaf",
            summary=parent_text,
            mid_offset=mid_offset,
            document_id="doc1",
        )

        # Create retrieval result with parent (only parent should be in frontier since only one child is covered)
        retrieval_result = RetrievalResult(
            node_ids=["root", "left_leaf"],
            scores={"root": 0.9, "left_leaf": 0.8},
            coverage_map={"root": True, "left_leaf": True},
            frontier_nodes=["root"],  # Only root in frontier since left_leaf is covered
        )

        # Assemble
        assembler = Assembler(config, store)
        result = assembler.assemble(retrieval_result)

        # Check for no duplicates
        assert result.count("Sarah discovered a book") == 1
        assert result.count("She went to the library") == 1
        assert "<<<MID>>>" not in result

    def test_span_consistency_in_tree(self, setup_components):
        """Test that parent spans equal union of child spans."""
        config, store, _, _ = setup_components

        # Add nodes with correct spans
        store.add_node(
            node_id="left",
            text="First half",
            embedding=[0.1] * 1536,
            depth=0,
            span_start=0,
            span_end=10,
            document_id="doc1",
        )

        store.add_node(
            node_id="right",
            text="Second half",
            embedding=[0.2] * 1536,
            depth=0,
            span_start=10,
            span_end=21,
            document_id="doc1",
        )

        store.add_node(
            node_id="parent",
            text="First half <<<MID>>> Second half",
            embedding=[0.15] * 1536,
            depth=1,
            span_start=0,
            span_end=21,
            left_child_id="left",
            right_child_id="right",
            summary="First half <<<MID>>> Second half",
            mid_offset=11,
            document_id="doc1",
        )

        # Verify spans
        parent = store.get_node("parent")
        left = store.get_node("left")
        right = store.get_node("right")

        assert parent.span_start == left.span_start
        assert parent.span_end == right.span_end
        assert left.span_end == right.span_start

    def test_mid_delimiter_extraction_no_overlaps(self, setup_components):
        """Test that mid-delimiter logic prevents content duplication."""
        config, store, _, _ = setup_components

        # Set up tree
        store.add_node(
            node_id="left_child",
            text="Chapter 1 content",
            embedding=[0.1] * 1536,
            depth=0,
            span_start=0,
            span_end=17,
            parent_id="parent",
            document_id="doc1",
        )

        store.add_node(
            node_id="right_child",
            text="Chapter 2 content",
            embedding=[0.2] * 1536,
            depth=0,
            span_start=17,
            span_end=34,
            parent_id="parent",
            document_id="doc1",
        )

        parent_text = "Chapter 1 summary <<<MID>>> Chapter 2 summary"
        store.add_node(
            node_id="parent",
            text=parent_text,
            embedding=[0.15] * 1536,
            depth=1,
            span_start=0,
            span_end=34,
            left_child_id="left_child",
            right_child_id="right_child",
            summary=parent_text,
            mid_offset=parent_text.find("<<<MID>>>"),
            document_id="doc1",
        )

        # Test Case 1: Parent + left child in frontier
        # NEW BEHAVIOR: Both are kept, parent outputs only right summary
        retrieval_result = RetrievalResult(
            node_ids=["parent", "left_child"],
            scores={"parent": 0.9, "left_child": 0.8},
            coverage_map={"parent": True, "left_child": True},
            frontier_nodes=["parent", "left_child"],
        )

        assembler = Assembler(config, store)
        result = assembler.assemble(retrieval_result)

        # Should get left child's full text + parent's right summary
        assert "Chapter 1 content" in result  # Full left child text
        assert "Chapter 2 summary" in result  # Parent's right summary
        assert "<<<MID>>>" not in result
        assert result.count("Chapter 1") == 1  # No duplication
        assert result.count("Chapter 2") == 1  # No duplication

    def test_slope_cap_deduplication(self, setup_components):
        """Test that slope cap doesn't create duplicate nodes."""
        config, store, _, _ = setup_components

        # Create nodes at different depths
        nodes = []
        for i in range(5):
            store.add_node(
                node_id=f"node_{i}",
                text=f"Content {i}",
                embedding=[i * 0.1] * 1536,
                depth=i % 3,  # Varying depths
                span_start=i * 10,
                span_end=(i + 1) * 10,
                document_id="doc1",
            )
            nodes.append(f"node_{i}")

        # Create a frontier with depth violations
        frontier = ["node_0", "node_4", "node_1", "node_4"]  # node_4 appears twice

        retrieval_result = RetrievalResult(
            node_ids=frontier,
            scores={n: 0.8 for n in frontier},
            coverage_map={n: True for n in frontier},
            frontier_nodes=frontier,
        )

        assembler = Assembler(config, store)

        # Apply slope cap through assembly
        result = assembler.assemble(retrieval_result)

        # Each content should appear only once
        assert result.count("Content 0") == 1
        assert result.count("Content 1") == 1
        assert result.count("Content 4") == 1

    def test_coverage_map_includes_ancestors(self, setup_components):
        """Test that coverage map includes all ancestors of frontier nodes."""
        config, store, _, _ = setup_components

        # Create a deeper tree
        store.add_node(
            node_id="leaf",
            text="Leaf content",
            embedding=[0.1] * 1536,
            depth=0,
            span_start=0,
            span_end=12,
            parent_id="middle",
            document_id="doc1",
        )

        store.add_node(
            node_id="middle",
            text="Middle content",
            embedding=[0.2] * 1536,
            depth=1,
            span_start=0,
            span_end=12,
            parent_id="root",
            document_id="doc1",
        )

        store.add_node(
            node_id="root",
            text="Root content",
            embedding=[0.3] * 1536,
            depth=2,
            span_start=0,
            span_end=12,
            document_id="doc1",
        )

        # Parent references are already set during node creation

        assembler = Assembler(config, store)
        coverage_map = assembler._build_coverage_map(["leaf"])

        # Coverage map should include leaf and all its ancestors
        assert "leaf" in coverage_map
        assert "middle" in coverage_map
        assert "root" in coverage_map

    def test_invalid_frontier_with_parent_and_child(self, setup_components):
        """Test handling when frontier incorrectly contains both parent and child."""
        config, store, mock_client, _ = setup_components

        # Create same tree as before
        store.add_node(
            node_id="left_leaf",
            text="Sarah discovered a book in the attic.",
            embedding=[0.1] * 1536,
            depth=0,
            span_start=0,
            span_end=38,
            parent_id="root",  # Set parent reference
            document_id="doc1",
        )

        store.add_node(
            node_id="right_leaf",
            text="She went to the library and met Mr. Chen.",
            embedding=[0.2] * 1536,
            depth=0,
            span_start=38,
            span_end=79,
            parent_id="root",  # Set parent reference
            document_id="doc1",
        )

        parent_text = "Sarah discovered a book in the attic. <<<MID>>> She went to the library and met Mr. Chen."
        store.add_node(
            node_id="root",
            text=parent_text,
            embedding=[0.15] * 1536,
            depth=1,
            span_start=0,
            span_end=79,
            left_child_id="left_leaf",
            right_child_id="right_leaf",
            summary=parent_text,
            mid_offset=parent_text.find("<<<MID>>>"),
            document_id="doc1",
        )

        # Create INVALID frontier with both parent and child
        retrieval_result = RetrievalResult(
            node_ids=["root", "left_leaf"],
            scores={"root": 0.9, "left_leaf": 0.8},
            coverage_map={"root": True, "left_leaf": True},
            frontier_nodes=["root", "left_leaf"],  # This is invalid!
        )

        # Even with invalid frontier, assembly should not duplicate content
        assembler = Assembler(config, store)
        result = assembler.assemble(retrieval_result)

        # Should still avoid duplicates through span deduplication
        assert result.count("Sarah discovered a book") == 1
        assert "<<<MID>>>" not in result

    def test_zero_width_span_handling(self, setup_components):
        """Test handling of zero-width spans (regression test)."""
        config, store, _, _ = setup_components

        # Create node with zero-width span (this was the bug)
        store.add_node(
            node_id="zero_width",
            text="",  # Empty text
            embedding=[0.1] * 1536,
            depth=1,
            span_start=100,
            span_end=100,  # Same as start!
            document_id="doc1",
        )

        store.add_node(
            node_id="normal",
            text="Normal content",
            embedding=[0.2] * 1536,
            depth=1,
            span_start=0,
            span_end=14,
            document_id="doc1",
        )

        retrieval_result = RetrievalResult(
            node_ids=["zero_width", "normal"],
            scores={"zero_width": 0.5, "normal": 0.9},
            coverage_map={"zero_width": True, "normal": True},
            frontier_nodes=["zero_width", "normal"],
        )

        assembler = Assembler(config, store)
        result = assembler.assemble(retrieval_result)

        # Should handle gracefully
        assert "Normal content" in result
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_full_pipeline_no_repetition(self, setup_components, tmp_path):
        """Test full indexing and retrieval pipeline produces no repetitions."""
        config, store, mock_client, mock_async_client = setup_components

        # Create test document
        test_doc = """Chapter 1: The Beginning
Sarah discovered an old book in her grandmother's attic.

Chapter 2: The Journey
She traveled to the library to decode the mysterious symbols.

Chapter 3: The End
The book revealed the location of a hidden treasure."""

        test_file = tmp_path / "test.txt"
        test_file.write_text(test_doc)

        # Mock embeddings - needs to be async
        async def mock_embeddings(*args, **kwargs):
            input_texts = kwargs.get("input", [])
            if isinstance(input_texts, str):
                input_texts = [input_texts]
            return MagicMock(
                data=[MagicMock(embedding=[0.1] * 1536) for _ in input_texts]
            )

        mock_async_client.embeddings.create.side_effect = mock_embeddings

        # Mock summaries with proper delimiters
        summaries = [
            "Sarah found a book. <<<MID>>> She went to the library.",
            "She decoded symbols. <<<MID>>> The treasure was revealed.",
            "Sarah's journey from discovery <<<MID>>> to finding treasure.",
        ]

        # Mock chat completions - needs to be async
        summary_index = 0

        async def mock_chat_completion(*args, **kwargs):
            nonlocal summary_index
            response = MagicMock()
            response.choices[0].message.content = summaries[
                summary_index % len(summaries)
            ]
            summary_index += 1
            return response

        mock_async_client.chat.completions.create.side_effect = mock_chat_completion

        # Index
        with patch("ragzoom.index.AsyncOpenAI", return_value=mock_async_client):
            builder = TreeBuilder(config, store)
            await builder.add_document_async(str(test_file))

        # Retrieve
        mock_client.embeddings.create.return_value = MagicMock(
            data=[MagicMock(embedding=[0.1] * 1536)]
        )

        with patch("ragzoom.retrieve.OpenAI", return_value=mock_client):
            retriever = Retriever(config, store)
            result = retriever.retrieve("Sarah book", n_max=10)

        # Assemble
        assembler = Assembler(config, store)
        final_text = assembler.assemble(result)

        # Verify no repetitions
        sentences = [s.strip() for s in final_text.split(".") if s.strip()]
        unique_sentences = set(sentences)

        # Each sentence should appear only once
        assert len(sentences) == len(
            unique_sentences
        ), f"Found duplicate sentences in output: {final_text}"

        # No mid delimiters in output
        assert "<<<MID>>>" not in final_text
