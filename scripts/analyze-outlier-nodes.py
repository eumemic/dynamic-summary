#!/usr/bin/env python3
"""Analyze bad summaries from indexing runs with distribution testing."""

import argparse
import asyncio
import json
import os
import sqlite3
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragzoom.config import IndexConfig, OperationalConfig, SecretStr
from ragzoom.index import TreeBuilder
from ragzoom.store import Store


@dataclass
class SummaryCase:
    """A summary case to analyze."""

    node_id: str
    height: int
    left_text: str
    right_text: str
    original_summary: str
    original_tokens: int
    target_tokens: int
    divergence: int
    divergence_pct: float
    preceding_context: str | None = None


class BadSummaryAnalyzer:
    """Analyze bad summaries with distribution testing."""

    def __init__(
        self,
        db_path: Path,
        telemetry_path: Path | None = None,
        target_tokens: int | None = None,
    ):
        """Initialize analyzer with database connection.

        Args:
            db_path: Path to the ragzoom.db database
            telemetry_path: Path to telemetry.json to read config from
            target_tokens: Optional override for target token count
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

        # Get API key from environment
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        # Check what columns the database has
        cursor = self.conn.execute("PRAGMA table_info(tree_nodes)")
        columns = [row[1] for row in cursor.fetchall()]
        self.has_token_count = "token_count" in columns
        self.has_preceding_neighbor = "preceding_neighbor_id" in columns

        # Load config from telemetry if available
        config_dict = None
        if telemetry_path and telemetry_path.exists():
            with open(telemetry_path) as f:
                telemetry = json.load(f)
            if "config" in telemetry:
                config_dict = telemetry["config"]
                print(f"Loaded config from telemetry: {telemetry_path}")

        # Override specific values if provided
        if target_tokens is not None:
            if config_dict:
                config_dict["target_chunk_tokens"] = target_tokens
            else:
                config_dict = {"target_chunk_tokens": target_tokens}
            print(f"Overriding target token count: {target_tokens}")

        # Create config using all values from telemetry (or defaults)
        if config_dict:
            # Use IndexConfig.from_dict to load all config values
            self.config = IndexConfig.from_dict(config_dict)
            self.target_tokens = self.config.target_chunk_tokens
            self.retry_threshold = self.config.retry_threshold
            print(
                f"Config: target={self.target_tokens} tokens, retry_threshold={self.retry_threshold}, model={self.config.summary_model}"
            )
        else:
            # Use defaults
            self.config = IndexConfig.load()
            self.target_tokens = self.config.target_chunk_tokens
            self.retry_threshold = self.config.retry_threshold
            print(
                f"Using default config: target={self.target_tokens} tokens, retry_threshold={self.retry_threshold}"
            )

        # Create store and TreeBuilder
        operational_config = OperationalConfig(
            openai_api_key=SecretStr(api_key),
            database_url="postgresql:///:memory:",
        )
        self.store = Store(
            operational_config, embedding_model=self.config.embedding_model
        )
        self.tree_builder = TreeBuilder(self.config, self.store, api_key=api_key)

    def get_worst_cases(self, top_n: int) -> list[SummaryCase]:
        """Get the worst N cases by absolute divergence that exceed retry threshold.

        Only returns cases that would have triggered retries:
        - Overshoots > 20% (retry_threshold)
        - Ignores undershoots (they're never retried per undershoot elimination)
        """
        if self.has_token_count:
            query = """
            SELECT 
                n.id as node_id,
                n.text as summary,
                n.token_count as original_tokens,
                n.left_child_id,
                n.right_child_id,
                n.preceding_neighbor_id,
                (n.token_count - ?) as divergence,
                ABS(n.token_count - ?) as abs_divergence,
                (n.token_count - ?) * 100.0 / ? as divergence_pct
            FROM tree_nodes n
            WHERE n.left_child_id IS NOT NULL  -- Non-leaf nodes only
              AND n.token_count IS NOT NULL
              AND n.token_count > ?  -- Only overshoots (undershoots are never retried)
              AND (n.token_count - ?) * 100.0 / ? > ?  -- Exceeds retry threshold
            ORDER BY abs_divergence DESC
            LIMIT ?
            """

            cursor = self.conn.execute(
                query,
                (
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.retry_threshold * 100,
                    top_n,
                ),
            )
        else:
            # Use text length as approximation (roughly 4 chars per token)
            query = """
            SELECT 
                n.id as node_id,
                n.text as summary,
                LENGTH(n.text) / 4 as original_tokens,
                n.left_child_id,
                n.right_child_id,
                n.preceding_neighbor_id,
                (LENGTH(n.text) / 4 - ?) as divergence,
                ABS(LENGTH(n.text) / 4 - ?) as abs_divergence,
                (LENGTH(n.text) / 4 - ?) * 100.0 / ? as divergence_pct
            FROM tree_nodes n
            WHERE n.left_child_id IS NOT NULL  -- Non-leaf nodes only
              AND n.text IS NOT NULL
              AND LENGTH(n.text) / 4 > ?  -- Only overshoots (undershoots are never retried)
              AND (LENGTH(n.text) / 4 - ?) * 100.0 / ? > ?  -- Exceeds retry threshold
            ORDER BY abs_divergence DESC
            LIMIT ?
            """

            cursor = self.conn.execute(
                query,
                (
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.retry_threshold * 100,
                    top_n,
                ),
            )

        cases = []
        for row in cursor.fetchall():
            # Get children for context
            left_child = self._get_node(row["left_child_id"])
            right_child = self._get_node(row["right_child_id"])

            if not left_child or not right_child:
                continue

            # Get preceding context if available
            preceding_context = None
            if row["preceding_neighbor_id"]:
                preceding_node = self._get_node(row["preceding_neighbor_id"])
                if preceding_node:
                    preceding_context = preceding_node["text"]

            # Calculate height
            height = self._calculate_height(row["node_id"])

            cases.append(
                SummaryCase(
                    node_id=row["node_id"],
                    height=height,
                    left_text=left_child["text"],
                    right_text=right_child["text"],
                    original_summary=row["summary"],
                    original_tokens=row["original_tokens"],
                    target_tokens=self.target_tokens,
                    divergence=row["divergence"],
                    divergence_pct=row["divergence_pct"],
                    preceding_context=preceding_context,
                )
            )

        return cases

    def get_specific_nodes(self, node_ids: list[str]) -> list[SummaryCase]:
        """Get specific nodes by ID."""
        cases = []

        for node_id in node_ids:
            cursor = self.conn.execute(
                "SELECT * FROM tree_nodes WHERE id = ?", (node_id,)
            )
            row = cursor.fetchone()

            if not row:
                print(f"Warning: Node {node_id} not found")
                continue

            # Get children for context
            left_child = self._get_node(row["left_child_id"])
            right_child = self._get_node(row["right_child_id"])

            if not left_child or not right_child:
                print(f"Warning: Node {node_id} is a leaf node, skipping")
                continue

            # Get preceding context if available
            preceding_context = None
            try:
                if row["preceding_neighbor_id"]:
                    preceding_node = self._get_node(row["preceding_neighbor_id"])
                    if preceding_node:
                        preceding_context = preceding_node["text"]
            except (KeyError, IndexError):
                # Column doesn't exist in this database
                pass

            # Calculate height and divergence
            height = self._calculate_height(node_id)
            if self.has_token_count:
                original_tokens = row["token_count"]
            else:
                original_tokens = len(row["text"]) // 4  # Approximate
            divergence = original_tokens - self.target_tokens
            divergence_pct = (
                (divergence / self.target_tokens) * 100 if self.target_tokens > 0 else 0
            )

            cases.append(
                SummaryCase(
                    node_id=node_id,
                    height=height,
                    left_text=left_child["text"],
                    right_text=right_child["text"],
                    original_summary=row["text"],
                    original_tokens=original_tokens,
                    target_tokens=self.target_tokens,
                    divergence=divergence,
                    divergence_pct=divergence_pct,
                    preceding_context=preceding_context,
                )
            )

        return cases

    def _get_node(self, node_id: str) -> sqlite3.Row | None:
        """Get node from database."""
        if not node_id:
            return None
        cursor = self.conn.execute("SELECT * FROM tree_nodes WHERE id = ?", (node_id,))
        return cursor.fetchone()

    def _calculate_height(self, node_id: str) -> int:
        """Calculate height of a node."""
        node = self._get_node(node_id)
        if not node or (not node["left_child_id"] and not node["right_child_id"]):
            return 0

        left_height = (
            self._calculate_height(node["left_child_id"])
            if node["left_child_id"]
            else 0
        )
        right_height = (
            self._calculate_height(node["right_child_id"])
            if node["right_child_id"]
            else 0
        )

        return 1 + max(left_height, right_height)

    async def _test_case_once(self, case: SummaryCase) -> dict:
        """Run a single test of a case."""
        summary, retry_count, final_tokens = await self.tree_builder._summarize_text(
            left_text=case.left_text,
            right_text=case.right_text,
            target_tokens=case.target_tokens,
            prev_context=case.preceding_context,
            parent_id=case.node_id,
        )

        # Check if result is verbatim
        combined_text = f"{case.left_text} {case.right_text}".strip()
        is_verbatim = summary.strip() == combined_text

        divergence = final_tokens - case.target_tokens
        divergence_pct = (
            (divergence / case.target_tokens) * 100 if case.target_tokens > 0 else 0
        )

        return {
            "tokens": final_tokens,
            "divergence": divergence,
            "divergence_pct": divergence_pct,
            "is_verbatim": is_verbatim,
            "summary": summary,
        }

    async def analyze_case_distribution(
        self, case: SummaryCase, n_runs: int = 10
    ) -> dict:
        """Analyze distribution for a single case with parallel runs."""
        # Run all tests in parallel
        tasks = [self._test_case_once(case) for _ in range(n_runs)]
        results = await asyncio.gather(*tasks)

        # Calculate statistics
        tokens = [r["tokens"] for r in results]
        divergences = [r["divergence"] for r in results]
        divergences_pct = [r["divergence_pct"] for r in results]
        verbatim_count = sum(1 for r in results if r["is_verbatim"])

        # Check for unique summaries
        summaries = [r["summary"] for r in results]
        unique_summaries = len(set(summaries))

        return {
            "case": case,
            "n_runs": n_runs,
            "results": results,
            "stats": {
                "mean_tokens": statistics.mean(tokens),
                "median_tokens": statistics.median(tokens),
                "min_tokens": min(tokens),
                "max_tokens": max(tokens),
                "range_tokens": max(tokens) - min(tokens),
                "stdev_tokens": statistics.stdev(tokens) if len(tokens) > 1 else 0,
                "mean_divergence_pct": statistics.mean(divergences_pct),
                "median_divergence_pct": statistics.median(divergences_pct),
                "stdev_divergence_pct": (
                    statistics.stdev(divergences_pct) if len(divergences_pct) > 1 else 0
                ),
                "verbatim_rate": (verbatim_count / n_runs) * 100,
                "unique_summaries": unique_summaries,
                "better_than_original": sum(
                    1 for d in divergences if abs(d) < abs(case.divergence)
                ),
                "better_than_original_pct": (
                    sum(1 for d in divergences if abs(d) < abs(case.divergence))
                    / n_runs
                )
                * 100,
            },
            "token_distribution": tokens,
        }

    def classify_problem(self, analysis: dict) -> str:
        """Classify the type of problem based on distribution."""
        stats = analysis["stats"]

        if stats["verbatim_rate"] > 50:
            return "SYSTEMATIC PROBLEM - High verbatim rate"
        elif stats["verbatim_rate"] > 20:
            return "SYSTEMATIC PROBLEM - Frequent verbatim"
        elif (
            stats["stdev_divergence_pct"] < 10
            and abs(stats["mean_divergence_pct"]) > 50
        ):
            return "SYSTEMATIC PROBLEM - Consistently bad"
        elif stats["better_than_original_pct"] > 80:
            return "OUTLIER - Original was statistical anomaly"
        elif stats["stdev_divergence_pct"] > 50:
            return "UNSTABLE - High variance"
        elif stats["range_tokens"] > 300:
            return "UNSTABLE - Bimodal distribution"
        else:
            return "MODERATE - May benefit from prompt tuning"

    def print_analysis(self, analysis: dict):
        """Print analysis for a single case."""
        case = analysis["case"]
        stats = analysis["stats"]

        # Calculate combined input size
        import tiktoken

        tokenizer = tiktoken.get_encoding("cl100k_base")
        left_tokens = len(tokenizer.encode(case.left_text))
        right_tokens = len(tokenizer.encode(case.right_text))
        combined_tokens = left_tokens + right_tokens
        compression_ratio = (
            (combined_tokens / case.target_tokens) if case.target_tokens > 0 else 0
        )

        print(f"\nNode {case.node_id} (height {case.height})")
        print(
            f"  Input: {combined_tokens} tokens ({left_tokens}+{right_tokens}) → target {case.target_tokens} ({compression_ratio:.1f}x compression)"
        )
        print(
            f"  Original: {case.original_tokens} tokens ({case.divergence:+d} from target)"
        )
        print(
            f"  Re-runs: {stats['min_tokens']}-{stats['max_tokens']} tokens "
            f"(mean: {stats['mean_tokens']:.0f}, median: {stats['median_tokens']:.0f})"
        )
        print(f"  Verbatim: {stats['verbatim_rate']:.0f}% of runs")
        print(f"  Assessment: {self.classify_problem(analysis)}")

    async def run_analysis(
        self, cases: list[SummaryCase], n_runs: int = 10, max_concurrent: int = 5
    ):
        """Run distribution analysis on all cases with controlled concurrency."""
        if not cases:
            print("No cases to analyze")
            return

        print(
            f"\nAnalyzing {len(cases)} summaries from {self.db_path} ({n_runs} runs each)..."
        )
        print(f"Target token count: {self.target_tokens}")

        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_semaphore(case: SummaryCase, idx: int) -> dict:
            async with semaphore:
                print(
                    f"\n[{idx+1}/{len(cases)}] Analyzing {case.node_id}...", flush=True
                )
                return await self.analyze_case_distribution(case, n_runs)

        # Run all analyses with controlled concurrency
        tasks = [analyze_with_semaphore(case, idx) for idx, case in enumerate(cases)]
        analyses = await asyncio.gather(*tasks)

        # Print results
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)

        systematic_problems = []
        outliers = []
        unstable = []

        for analysis in analyses:
            self.print_analysis(analysis)
            classification = self.classify_problem(analysis)

            if "SYSTEMATIC PROBLEM" in classification:
                systematic_problems.append(analysis["case"].node_id)
            elif "OUTLIER" in classification:
                outliers.append(analysis["case"].node_id)
            elif "UNSTABLE" in classification:
                unstable.append(analysis["case"].node_id)

        # Print summary
        print("\n" + "-" * 60)
        print(
            f"Summary: {len(systematic_problems)} systematic problems, "
            f"{len(unstable)} unstable, {len(outliers)} outliers"
        )

        if systematic_problems:
            print("\nSystematic problems (focus on these):")
            for node_id in systematic_problems:
                print(f"  - {node_id}")

        if unstable:
            print("\nUnstable cases:")
            for node_id in unstable:
                print(f"  - {node_id}")


async def main():
    parser = argparse.ArgumentParser(
        description="Analyze bad summaries with distribution testing"
    )

    # Default to benchmarks/latest if it exists
    default_db = (
        Path("benchmarks/latest/ragzoom.db")
        if Path("benchmarks/latest/ragzoom.db").exists()
        else Path("./ragzoom.db")
    )

    parser.add_argument(
        "--db",
        type=Path,
        default=default_db,
        help=f"Database path (default: {default_db})",
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=None,
        help="Path to telemetry.json to read config from (default: auto-detect from db path)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of worst cases to analyze (default: 10)",
    )
    parser.add_argument(
        "--nodes",
        type=str,
        help="Specific node IDs to analyze (comma-separated, overrides --top)",
    )
    parser.add_argument(
        "--target", type=int, help="Target token count (overrides value from telemetry)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of test runs per case (default: 10)",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=5,
        help="Maximum concurrent analyses (default: 5)",
    )
    parser.add_argument(
        "--show-text", action="store_true", help="Show input and output text for cases"
    )

    args = parser.parse_args()

    if not args.db.exists():
        print(f"Error: Database {args.db} does not exist")
        sys.exit(1)

    # Auto-detect telemetry path if not provided
    telemetry_path = args.telemetry
    if telemetry_path is None:
        # Try to find telemetry.json in the same directory as the database
        potential_telemetry = args.db.parent / "telemetry.json"
        if potential_telemetry.exists():
            telemetry_path = potential_telemetry
            print(f"Auto-detected telemetry: {telemetry_path}")

    # Create analyzer
    analyzer = BadSummaryAnalyzer(
        args.db, telemetry_path=telemetry_path, target_tokens=args.target
    )

    # Get cases to analyze
    if args.nodes:
        # Specific nodes mode
        node_ids = [nid.strip() for nid in args.nodes.split(",")]
        cases = analyzer.get_specific_nodes(node_ids)
        print(f"Analyzing specific nodes: {', '.join(node_ids)}")
    else:
        # Top N mode
        cases = analyzer.get_worst_cases(args.top)
        print(f"Found {len(cases)} worst cases by divergence")

    # Run analysis
    await analyzer.run_analysis(
        cases, n_runs=args.runs, max_concurrent=args.max_concurrent
    )

    # Show detailed text if requested
    if args.show_text and cases:
        print("\n" + "=" * 60)
        print("DETAILED TEXT ANALYSIS")
        print("=" * 60)

        for case in cases[:3]:  # Show first 3 cases
            import tiktoken

            tokenizer = tiktoken.get_encoding("cl100k_base")

            print(f"\n### Node {case.node_id} (height {case.height})")
            print(
                f"Target: {case.target_tokens} tokens, Original output: {case.original_tokens} tokens"
            )
            print(
                f"\n--- LEFT INPUT ({len(tokenizer.encode(case.left_text))} tokens) ---"
            )
            print(
                case.left_text[:500] + "..."
                if len(case.left_text) > 500
                else case.left_text
            )
            print(
                f"\n--- RIGHT INPUT ({len(tokenizer.encode(case.right_text))} tokens) ---"
            )
            print(
                case.right_text[:500] + "..."
                if len(case.right_text) > 500
                else case.right_text
            )
            print(f"\n--- ORIGINAL SUMMARY ({case.original_tokens} tokens) ---")
            print(case.original_summary)
            print("\n" + "-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
