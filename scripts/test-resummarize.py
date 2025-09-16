#!/usr/bin/env python3
"""
Test re-summarization of a specific node without modifying the database.
Shows left/right text, old summary, new summary, and token counts.
"""

import argparse
import asyncio
import os
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ruff: noqa: E402
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragzoom.config import IndexConfig
from ragzoom.index import TreeBuilder
from ragzoom.store import Store


def get_node_data(node_id: str, db_path: str = "ragzoom.db"):
    """Fetch node data from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get the parent node
    cursor.execute(
        """
        SELECT id, text, token_count, left_child_id, right_child_id
        FROM tree_nodes
        WHERE id = ?
    """,
        (node_id,),
    )

    parent = cursor.fetchone()
    if not parent:
        print(f"❌ Node {node_id} not found")
        return None

    parent_id, parent_text, parent_tokens, left_id, right_id = parent

    # Get left child
    cursor.execute(
        """
        SELECT text, token_count
        FROM tree_nodes
        WHERE id = ?
    """,
        (left_id,),
    )
    left = cursor.fetchone()

    # Get right child
    cursor.execute(
        """
        SELECT text, token_count
        FROM tree_nodes
        WHERE id = ?
        """,
        (right_id,),
    )
    right = cursor.fetchone()

    conn.close()

    if not left or not right:
        print(f"❌ Could not find children for node {node_id}")
        return None

    return {
        "parent_id": parent_id,
        "parent_text": parent_text,
        "parent_tokens": parent_tokens,
        "left_text": left[0],
        "left_tokens": left[1],
        "right_text": right[0],
        "right_tokens": right[1],
    }


async def test_resummarize(node_id: str, target_tokens: int = 200):
    """Re-run summarization for a specific node."""

    # Get node data
    data = get_node_data(node_id)
    if not data:
        return

    print("=" * 80)
    print(f"TESTING NODE: {node_id}")
    print("=" * 80)

    print("\n📊 TOKEN COUNTS:")
    print(f"  Left child:  {data['left_tokens']} tokens")
    print(f"  Right child: {data['right_tokens']} tokens")
    print(f"  Combined:    {data['left_tokens'] + data['right_tokens']} tokens")
    print(f"  Old summary: {data['parent_tokens']} tokens")
    print(f"  Target:      {target_tokens} tokens")

    print(f"\n📝 LEFT CHILD TEXT ({data['left_tokens']} tokens):")
    print("-" * 40)
    print(data["left_text"][:500] + ("..." if len(data["left_text"]) > 500 else ""))

    print(f"\n📝 RIGHT CHILD TEXT ({data['right_tokens']} tokens):")
    print("-" * 40)
    print(data["right_text"][:500] + ("..." if len(data["right_text"]) > 500 else ""))

    print(f"\n📝 OLD SUMMARY ({data['parent_tokens']} tokens):")
    print("-" * 40)
    print(data["parent_text"])

    # Re-run summarization using actual TreeBuilder
    print("\n🔄 RE-RUNNING SUMMARIZATION...")
    print("-" * 40)

    try:
        # Load the actual config used during indexing
        config = IndexConfig.load()

        # Allow overriding target_tokens for testing
        if target_tokens != config.target_chunk_tokens:
            print(
                f"\n⚠️  Note: Overriding target_tokens from {config.target_chunk_tokens} to {target_tokens}"
            )
            config = config.replace(target_chunk_tokens=target_tokens)

        # Show which model is being used
        print(f"\n📊 Using model: {config.summary_model}")

        # Create a temporary store (we won't save to it)
        from ragzoom.config import OperationalConfig

        op_config = OperationalConfig(database_url="postgresql://temp")
        store = Store(op_config, embedding_model=config.embedding_model)

        # Create tree builder
        builder = TreeBuilder(
            config=config, store=store, api_key=os.getenv("OPENAI_API_KEY")
        )

        # Re-run summarization
        import time

        start_time = time.time()
        new_summary, retries, output_tokens = await builder._summarize_text(
            data["left_text"],
            data["right_text"],
            parent_id=node_id,
            target_tokens=target_tokens,
        )
        elapsed = time.time() - start_time

        print(f"\n✅ NEW SUMMARY ({output_tokens} tokens, {elapsed:.2f}s):")
        print("-" * 40)
        print(new_summary)

        # Compare
        print("\n📊 COMPARISON:")
        print(f"  Old tokens: {data['parent_tokens']}")
        print(f"  New tokens: {output_tokens}")
        print(f"  Difference: {output_tokens - data['parent_tokens']} tokens")
        print(
            f"  Compression ratio: {(data['left_tokens'] + data['right_tokens']) / output_tokens:.2f}x"
        )
        print(f"  Retries needed: {retries}")

        # Check if it's verbatim
        combined = data["left_text"] + " " + data["right_text"]
        combined_no_space = data["left_text"] + data["right_text"]
        combined_single_para = data["left_text"] + " " + data["right_text"]

        # Also check if it's just concatenated with paragraph break removed
        if "\n" in data["left_text"]:
            last_para_break = data["left_text"].rfind("\n")
            combined_merged_para = (
                data["left_text"][:last_para_break]
                + " "
                + data["left_text"][last_para_break + 1 :]
                + " "
                + data["right_text"]
            )
        else:
            combined_merged_para = None

        if new_summary == combined:
            print("\n⚠️  WARNING: New summary is EXACT concatenation with space!")
        elif new_summary == combined_no_space:
            print("\n⚠️  WARNING: New summary is EXACT concatenation without space!")
        elif new_summary == data["parent_text"]:
            print("\n⚠️  WARNING: New summary is IDENTICAL to old summary!")
        elif new_summary == combined_single_para:
            print("\n⚠️  WARNING: New summary is concatenation as single paragraph!")
        elif combined_merged_para and new_summary == combined_merged_para:
            print(
                "\n⚠️  WARNING: New summary is concatenation with paragraph breaks merged!"
            )
        else:
            # Check similarity
            import difflib

            similarity = difflib.SequenceMatcher(None, combined, new_summary).ratio()
            if similarity > 0.95:
                print(
                    f"\n⚠️  WARNING: New summary is {similarity*100:.1f}% similar to concatenation!"
                )
            else:
                print(
                    f"\n✅ Summary appears to be properly compressed (similarity: {similarity*100:.1f}%)"
                )

        # Clean up
        # No cleanup methods needed

        # Remove temp database
        if os.path.exists("temp.db"):
            os.remove("temp.db")

    except Exception as e:
        print(f"\n❌ Error during summarization: {e}")
        import traceback

        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Test re-summarization of a specific node"
    )
    parser.add_argument("node_id", help="ID of the node to re-summarize")
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=200,
        help="Target token count (default: 200)",
    )
    parser.add_argument(
        "--db", default="ragzoom.db", help="Database path (default: ragzoom.db)"
    )

    args = parser.parse_args()

    asyncio.run(test_resummarize(args.node_id, args.target_tokens))


if __name__ == "__main__":
    main()
