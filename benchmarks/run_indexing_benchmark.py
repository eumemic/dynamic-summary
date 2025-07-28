#!/usr/bin/env python3
"""
Standalone benchmark script for RagZoom indexing performance.

This script benchmarks the indexing performance using real documents
from the test_data directory at various chunk sizes.

Usage:
    python benchmarks/run_indexing_benchmark.py [--chunk-sizes 100,200,400] [--output-dir benchmark_results]
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add parent directory to path to import ragzoom
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragzoom.config import RagZoomConfig
from ragzoom.index import TreeBuilder
from ragzoom.metrics import IndexingMetrics
from ragzoom.store import Store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs indexing benchmarks with real documents."""

    # Document configurations - chosen for variety and realistic use cases
    DOCUMENTS = {
        "narrative": {
            "path": "test_data/the_hobbit_chapter_1.txt",
            "name": "The Hobbit Chapter 1",
            "description": "Narrative prose (~7K tokens)",
        },
        "technical": {
            "path": "test_data/smoke_test_larger.txt",
            "name": "Technical Documentation",
            "description": "Technical content with code",
        },
        "classic": {
            "path": "test_data/moby_dick_sample.txt",
            "name": "Moby Dick Sample",
            "description": "Classic literature",
        },
    }

    def __init__(self, chunk_sizes: list[int], output_dir: Path):
        self.chunk_sizes = chunk_sizes
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)

    def run_benchmark(
        self, config: RagZoomConfig, document_path: Path, document_name: str
    ) -> dict[int, IndexingMetrics]:
        """Run benchmark for a single document at all chunk sizes."""
        # Read document
        text = document_path.read_text(encoding="utf-8")
        logger.info(f"Loaded {document_name}: {len(text)} characters")

        results = {}

        for chunk_size in self.chunk_sizes:
            logger.info(f"\n{'='*60}")
            logger.info(f"Benchmarking {document_name} with {chunk_size} token chunks")
            logger.info(f"{'='*60}")

            # Update config
            config.leaf_tokens = chunk_size

            # Create fresh store for each run
            with Store.temporary() as store:
                builder = TreeBuilder(config, store)

                # Warm up tokenizer
                _ = builder.splitter.tokenizer.encode("warmup")

                # Run indexing with metrics
                start_time = time.time()
                doc_id, metrics = builder.add_document_with_metrics(
                    text,
                    document_id=f"{document_name}_{chunk_size}",
                    show_progress=False,
                )
                end_time = time.time()

                # Verify timing
                actual_duration = end_time - start_time
                assert abs(metrics.total_duration_seconds - actual_duration) < 0.1

                results[chunk_size] = metrics

                # Log summary
                logger.info(f"✅ Completed in {metrics.total_duration_seconds:.2f}s")
                logger.info(
                    f"   Throughput: {metrics.tokens_per_second:.1f} tokens/sec"
                )
                logger.info(
                    f"   Cost per 1K tokens: ${metrics.cost_per_1k_tokens:.4f}"
                )
                logger.info(
                    f"   Peak memory: {metrics.peak_memory_mb:.1f} MB"
                )

        return results

    def save_results(
        self, results: dict[str, dict[int, IndexingMetrics]], timestamp: float
    ) -> None:
        """Save benchmark results to JSON files."""
        # Save individual metrics files (for compatibility with CI)
        for chunk_size in self.chunk_sizes:
            # Aggregate metrics across documents
            aggregated = self._aggregate_metrics(
                {doc: metrics[chunk_size] for doc, metrics in results.items()}
            )

            output_file = self.output_dir / f"metrics_{chunk_size}_tokens.json"
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "config": {
                            "leaf_tokens": chunk_size,
                            "documents": list(results.keys()),
                        },
                        "metrics": aggregated,
                        "timestamp": timestamp,
                    },
                    f,
                    indent=2,
                )
            logger.info(f"Saved {output_file}")

        # Save detailed results
        detailed_file = self.output_dir / "detailed_results.json"
        detailed_data = {}
        for doc_type, metrics_by_size in results.items():
            detailed_data[doc_type] = {
                str(size): metrics.to_dict() for size, metrics in metrics_by_size.items()
            }

        with open(detailed_file, "w") as f:
            json.dump(
                {
                    "documents": self.DOCUMENTS,
                    "results": detailed_data,
                    "timestamp": timestamp,
                },
                f,
                indent=2,
            )
        logger.info(f"Saved detailed results to {detailed_file}")

    def _aggregate_metrics(self, metrics_by_doc: dict[str, IndexingMetrics]) -> dict:
        """Aggregate metrics across multiple documents."""
        # For now, average the key metrics
        total_docs = len(metrics_by_doc)

        aggregated = {
            "timing": {
                "total_duration_seconds": 0,
                "tokens_per_second": 0,
                "time_per_1k_tokens": 0,
            },
            "document": {
                "source_tokens": 0,
                "chunks_created": 0,
            },
            "api_usage": {
                "total_calls": 0,
                "embedding_calls": 0,
                "summary_calls": 0,
                "embedding_tokens": 0,
                "summary_prompt_tokens": 0,
                "summary_completion_tokens": 0,
            },
            "efficiency": {
                "avg_embedding_batch_size": 0,
                "embedding_tokens_per_1k": 0,
                "summary_tokens_per_1k": 0,
                "api_calls_per_1k": 0,
                "cost_per_1k_tokens": 0,
            },
        }

        for doc, metrics in metrics_by_doc.items():
            m_dict = metrics.to_dict()

            # Add to aggregated values
            for category in aggregated:
                for key in aggregated[category]:
                    if key in m_dict.get(category, {}):
                        aggregated[category][key] += m_dict[category][key]

        # Average the values
        for category in aggregated:
            for key in aggregated[category]:
                aggregated[category][key] /= total_docs

        return aggregated

    def run_all_benchmarks(self, config: RagZoomConfig) -> None:
        """Run benchmarks for all documents."""
        timestamp = time.time()
        all_results = {}

        for doc_type, doc_info in self.DOCUMENTS.items():
            doc_path = Path(doc_info["path"])
            if not doc_path.exists():
                logger.warning(f"Skipping {doc_info['name']}: {doc_path} not found")
                continue

            logger.info(f"\n{'#'*80}")
            logger.info(f"# Benchmarking: {doc_info['name']}")
            logger.info(f"# {doc_info['description']}")
            logger.info(f"{'#'*80}")

            results = self.run_benchmark(config, doc_path, doc_type)
            all_results[doc_type] = results

        # Save results
        self.save_results(all_results, timestamp)

        # Print summary
        self.print_summary(all_results)

    def print_summary(self, results: dict[str, dict[int, IndexingMetrics]]) -> None:
        """Print a summary table of results."""
        logger.info(f"\n{'='*80}")
        logger.info("BENCHMARK SUMMARY")
        logger.info(f"{'='*80}")

        # Print header
        header = f"{'Document':<15} {'Chunk':<8} {'Tokens/s':<12} {'Time/1K':<10} {'Cost/1K':<10} {'Memory':<10}"
        logger.info(header)
        logger.info("-" * len(header))

        # Print results
        for doc_type, metrics_by_size in results.items():
            doc_name = self.DOCUMENTS[doc_type]["name"][:15]
            for chunk_size, metrics in sorted(metrics_by_size.items()):
                logger.info(
                    f"{doc_name:<15} {chunk_size:<8} "
                    f"{metrics.tokens_per_second:<12.1f} "
                    f"{metrics.time_per_1k_tokens:<10.2f} "
                    f"${metrics.cost_per_1k_tokens:<9.4f} "
                    f"{metrics.peak_memory_mb:<10.1f}"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Run RagZoom indexing performance benchmarks"
    )
    parser.add_argument(
        "--chunk-sizes",
        type=str,
        default="100,200,400",
        help="Comma-separated list of chunk sizes to test (default: 100,200,400)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for results (default: benchmark_results)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (or set RAGZOOM_OPENAI_API_KEY environment variable)",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="text-embedding-3-small",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--summary-model",
        type=str,
        default="gpt-4o-mini",
        help="Summary model to use",
    )

    args = parser.parse_args()

    # Parse chunk sizes
    chunk_sizes = [int(s.strip()) for s in args.chunk_sizes.split(",")]

    # Create config
    config = RagZoomConfig(
        openai_api_key=args.api_key,
        embedding_model=args.embedding_model,
        summary_model=args.summary_model,
        embedding_batch_size=100,
    )

    # Check API key
    if not config.openai_api_key or config.openai_api_key == "test-key":
        logger.error("No OpenAI API key provided. Set RAGZOOM_OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Run benchmarks
    runner = BenchmarkRunner(chunk_sizes, args.output_dir)
    runner.run_all_benchmarks(config)


if __name__ == "__main__":
    main()
