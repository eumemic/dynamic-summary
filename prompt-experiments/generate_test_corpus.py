#!/usr/bin/env python3
"""
Generate test corpus by splitting the_hobbit.txt at different chunk sizes.
This creates a diverse set of text chunks for testing summarization strategies.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import tiktoken

from ragzoom.config import IndexConfig
from ragzoom.splitter import TextSplitter


def generate_corpus():
    """Generate test corpus from the_hobbit.txt at multiple chunk sizes."""

    # Load the source text
    source_file = Path("test_data/the_hobbit.txt")
    if not source_file.exists():
        print(f"Error: {source_file} not found")
        return

    with open(source_file, encoding="utf-8") as f:
        text = f.read()

    print(f"Loaded {source_file.name}: {len(text)} characters")

    # Initialize tokenizer for counting
    tokenizer = tiktoken.get_encoding("cl100k_base")

    # Chunk sizes to test
    chunk_sizes = [200, 500, 1000]

    all_chunks = []
    chunk_id = 0

    for chunk_size in chunk_sizes:
        print(f"\nGenerating chunks of size {chunk_size} tokens...")

        # Create config with this chunk size
        config = IndexConfig(
            target_chunk_tokens=chunk_size,
            max_parallelism=30,
            summary_model="gpt-5-nano",
            embedding_model="text-embedding-3-small",
            retry_threshold=0.2,
            max_retries=0,
            embedding_batch_size=100,
            use_anti_verbatim_vaccine=True,
            processing_strategy="bottom_to_top",
        )

        # Create splitter
        splitter = TextSplitter(config)

        # Split text
        chunks = splitter.split_text(text)

        print(f"  Generated {len(chunks)} chunks")

        # Process each chunk
        for i, chunk_text in enumerate(chunks):
            # Calculate metrics
            tokens = tokenizer.encode(chunk_text)
            token_count = len(tokens)
            char_count = len(chunk_text)
            word_count = len(chunk_text.split())

            # Store chunk data
            chunk_data = {
                "id": f"chunk_{chunk_id:04d}",
                "source": "the_hobbit.txt",
                "target_chunk_size": chunk_size,
                "chunk_index": i,
                "text": chunk_text,
                "metrics": {
                    "tokens": token_count,
                    "characters": char_count,
                    "words": word_count,
                },
            }

            all_chunks.append(chunk_data)
            chunk_id += 1

            # Show sample info for first few chunks
            if i < 3:
                print(
                    f"    Chunk {i}: {token_count} tokens, {char_count} chars, {word_count} words"
                )

    # Save corpus
    output_file = Path("experiments/results/corpus.json")
    output_file.parent.mkdir(exist_ok=True, parents=True)

    corpus = {
        "source_file": str(source_file),
        "chunk_sizes": chunk_sizes,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Corpus saved to {output_file}")
    print(f"   Total chunks: {len(all_chunks)}")

    # Print distribution summary
    print("\nChunk distribution:")
    for size in chunk_sizes:
        count = sum(1 for c in all_chunks if c["target_chunk_size"] == size)
        print(f"  {size} tokens: {count} chunks")


if __name__ == "__main__":
    generate_corpus()
