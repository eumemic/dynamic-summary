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
from typing import Any

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
        # For now, average most metrics but use max for peak memory
        total_docs = len(metrics_by_doc)

        aggregated: dict[str, Any] = {
            "timing": {
                "total_duration_seconds": 0.0,
                "tokens_per_second": 0.0,
                "time_per_1k_tokens": 0.0,
            },
            "document": {
                "source_tokens": 0.0,
                "chunks_created": 0.0,
            },
            "api_usage": {
                "total_calls": 0.0,
                "embedding_calls": 0.0,
                "summary_calls": 0.0,
                "embedding_tokens": 0.0,
                "summary_prompt_tokens": 0.0,
                "summary_completion_tokens": 0.0,
            },
            "efficiency": {
                "avg_embedding_batch_size": 0.0,
                "embedding_tokens_per_1k": 0.0,
                "summary_tokens_per_1k": 0.0,
                "api_calls_per_1k": 0.0,
                "cost_per_1k_tokens": 0.0,
            },
            "memory": {
                "peak_mb": 0.0,
                "start_mb": 0.0,
                "end_mb": 0.0,
                "usage_mb": 0.0,
            },
            "amplification": {
                "median_cost": 0.0,
                "cost_p90": 0.0,
                "cost_p95": 0.0,
                "median_input": 0.0,
                "median_output": 0.0,
            },
        }

        # Collect all values for aggregation
        memory_peaks = []
        memory_starts = []
        memory_ends = []
        memory_usages = []

        # Collect summary accuracy data by target size
        summary_accuracy_by_target: dict[str, list[dict[str, Any]]] = {}

        # Collect raw deviation values for proper percentile calculation
        raw_deviations_by_target: dict[str, list[float]] = {}

        # Collect raw amplification values for proper percentile calculation
        all_cost_amplifications: list[float] = []
        all_input_amplifications: list[float] = []
        all_output_amplifications: list[float] = []

        for doc, metrics in metrics_by_doc.items():
            m_dict = metrics.to_dict()

            # Add to aggregated values (for averaging)
            for category in aggregated:
                if category == "memory":
                    # Collect memory values for special handling
                    if "memory" in m_dict:
                        memory_peaks.append(m_dict["memory"].get("peak_mb", 0))
                        memory_starts.append(m_dict["memory"].get("start_mb", 0))
                        memory_ends.append(m_dict["memory"].get("end_mb", 0))
                        memory_usages.append(m_dict["memory"].get("usage_mb", 0))
                else:
                    for key in aggregated[category]:
                        if key in m_dict.get(category, {}):
                            aggregated[category][key] += m_dict[category][key]

            # Collect summary accuracy data
            if "summary_accuracy" in m_dict:
                for target_size, stats in m_dict["summary_accuracy"].items():
                    if target_size not in summary_accuracy_by_target:
                        summary_accuracy_by_target[target_size] = []
                    summary_accuracy_by_target[target_size].append(stats)

                    # Collect raw deviations if available
                    if target_size not in raw_deviations_by_target:
                        raw_deviations_by_target[target_size] = []
                    if "deviations" in stats and stats["deviations"]:
                        raw_deviations_by_target[target_size].extend(stats["deviations"])

            # Collect raw amplification values
            if "amplification" in m_dict:
                if "cost_amplifications" in metrics.__dict__:
                    all_cost_amplifications.extend(metrics.cost_amplifications)
                if "input_amplifications" in metrics.__dict__:
                    all_input_amplifications.extend(metrics.input_amplifications)
                if "output_amplifications" in metrics.__dict__:
                    all_output_amplifications.extend(metrics.output_amplifications)

        # Average the non-memory values
        for category in aggregated:
            if category != "memory":
                for key in aggregated[category]:
                    aggregated[category][key] /= total_docs

        # Handle memory metrics specially
        if memory_peaks:
            aggregated["memory"]["peak_mb"] = max(memory_peaks)  # Use max for peak
            aggregated["memory"]["start_mb"] = sum(memory_starts) / len(memory_starts)  # Average start
            aggregated["memory"]["end_mb"] = sum(memory_ends) / len(memory_ends)  # Average end
            aggregated["memory"]["usage_mb"] = max(memory_usages)  # Use max for usage

        # Aggregate amplification metrics from raw values
        if all_cost_amplifications:
            import statistics
            aggregated["amplification"]["median_cost"] = statistics.median(all_cost_amplifications)
            sorted_costs = sorted(all_cost_amplifications)
            aggregated["amplification"]["cost_p90"] = sorted_costs[int(len(sorted_costs) * 0.9)]
            aggregated["amplification"]["cost_p95"] = sorted_costs[int(len(sorted_costs) * 0.95)]

        if all_input_amplifications:
            aggregated["amplification"]["median_input"] = statistics.median(all_input_amplifications)

        if all_output_amplifications:
            aggregated["amplification"]["median_output"] = statistics.median(all_output_amplifications)

        # Aggregate summary accuracy stats
        if summary_accuracy_by_target:
            aggregated["summary_accuracy"] = {}
            for target_size, stats_list in summary_accuracy_by_target.items():
                # Aggregate stats for this target size
                total_count = sum(s["count"] for s in stats_list)
                total_tokens = sum(s["avg_tokens"] * s["count"] for s in stats_list)

                # Combine histogram buckets
                combined_histogram = {}
                for bucket in ["0-10%", "10-25%", "25-50%", "50-100%", "100%+"]:
                    bucket_count = sum(s["histogram"].get(bucket, {}).get("count", 0) for s in stats_list)
                    bucket_percentage = (bucket_count / total_count * 100) if total_count > 0 else 0
                    combined_histogram[bucket] = {
                        "count": bucket_count,
                        "percentage": bucket_percentage
                    }

                # Calculate percentiles from raw values if available
                if target_size in raw_deviations_by_target and raw_deviations_by_target[target_size]:
                    import statistics
                    raw_devs = sorted(raw_deviations_by_target[target_size])
                    median_deviation = statistics.median(raw_devs)
                    std_deviation = statistics.stdev(raw_devs) if len(raw_devs) > 1 else 0
                    percentile_50 = raw_devs[int(len(raw_devs) * 0.5)]
                    percentile_90 = raw_devs[int(len(raw_devs) * 0.9)] if len(raw_devs) > 1 else percentile_50
                    percentile_95 = raw_devs[int(len(raw_devs) * 0.95)] if len(raw_devs) > 1 else percentile_50
                else:
                    # Fallback to averaging if raw values not available
                    median_deviation = sum(s.get("median_deviation_percent", 0) for s in stats_list) / len(stats_list)
                    std_deviation = sum(s.get("std_deviation_percent", 0) for s in stats_list) / len(stats_list)
                    percentile_50 = sum(s.get("percentile_50", 0) for s in stats_list) / len(stats_list)
                    percentile_90 = sum(s.get("percentile_90", 0) for s in stats_list) / len(stats_list)
                    percentile_95 = sum(s.get("percentile_95", 0) for s in stats_list) / len(stats_list)

                aggregated["summary_accuracy"][target_size] = {
                    "count": total_count,
                    "avg_tokens": total_tokens / total_count if total_count > 0 else 0,
                    "avg_deviation_percent": sum(s["avg_deviation_percent"] * s["count"] for s in stats_list) / total_count if total_count > 0 else 0,
                    "median_deviation_percent": median_deviation,
                    "std_deviation_percent": std_deviation,
                    "percentile_50": percentile_50,
                    "percentile_90": percentile_90,
                    "percentile_95": percentile_95,
                    "percent_over_target": sum(s["percent_over_target"] for s in stats_list) / len(stats_list),
                    "percent_under_target": sum(s["percent_under_target"] for s in stats_list) / len(stats_list),
                    "max_overage_percent": max(s["max_overage_percent"] for s in stats_list),
                    "max_underage_percent": max(s["max_underage_percent"] for s in stats_list),
                    "histogram": combined_histogram
                }

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

        # Print detailed summary accuracy stats
        self.print_summary_accuracy_details(results)

    def print_summary_accuracy_details(self, results: dict[str, dict[int, IndexingMetrics]]) -> None:
        """Print detailed summary accuracy statistics."""
        logger.info(f"\n{'='*80}")
        logger.info("SUMMARY ACCURACY DETAILS")
        logger.info(f"{'='*80}")

        for doc_type, metrics_by_size in results.items():
            for chunk_size, metrics in sorted(metrics_by_size.items()):
                if not metrics.summary_stats:
                    continue

                for target, stats in metrics.summary_stats.items():
                    if stats.count == 0:
                        continue

                    logger.info(f"\n📏 {self.DOCUMENTS[doc_type]['name']} - Target {target} tokens:")
                    logger.info(f"  Count: {stats.count}")
                    logger.info(f"  Average size: {stats.avg_tokens:.1f} tokens")
                    logger.info(f"  Average deviation: {stats.avg_deviation_percent:.1f}%")
                    logger.info(f"  Median deviation: {stats.median_deviation_percent:.1f}%")
                    logger.info(f"  Std deviation: {stats.std_deviation_percent:.1f}%")
                    logger.info(f"  Over target: {stats.percent_over_target:.1f}% (max: {stats.max_overage_percent:.1f}%)")
                    logger.info(f"  Under target: {stats.percent_under_target:.1f}% (max: {stats.max_underage_percent:.1f}%)")

                    # Percentiles
                    logger.info("\n  Percentiles:")
                    logger.info(f"    P50 (median): {stats.percentile_50:.1f}%")
                    logger.info(f"    P90: {stats.percentile_90:.1f}%")
                    logger.info(f"    P95: {stats.percentile_95:.1f}%")

                    # Histogram
                    logger.info("\n  Distribution:")
                    for bucket, data in stats.histogram.items():
                        bar_length = int(data['percentage'] / 2)  # Scale to fit
                        bar = "█" * bar_length
                        logger.info(f"    {bucket:>8}: {bar:<50} {data['count']:3d} ({data['percentage']:5.1f}%)")


def main() -> None:
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
    if not config.openai_api_key or config.openai_api_key in ["test-key", ""]:
        logger.error("No OpenAI API key provided. Set RAGZOOM_OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)

    # Run benchmarks
    runner = BenchmarkRunner(chunk_sizes, args.output_dir)
    runner.run_all_benchmarks(config)


if __name__ == "__main__":
    main()
