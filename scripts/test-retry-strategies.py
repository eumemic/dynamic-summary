#!/usr/bin/env python3
"""
Experiment to compare retry strategies for summarization:
1. Continuation: Continue conversation with corrective prompt (current approach)
2. Fresh: Start a new conversation with original prompt

Uses outliers from database as test cases and runs multiple trials per example.
"""

import asyncio
import json
import os
import sqlite3
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import click
import tiktoken
from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

# ruff: noqa: E402
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Constants from ragzoom/index.py
WORDS_PER_TOKEN = 0.75 * 0.94  # 0.705
MODEL = "gpt-5-nano"


@dataclass
class SummaryCase:
    """A summary case to test."""

    node_id: str
    left_text: str
    right_text: str
    original_summary: str
    original_tokens: int
    target_tokens: int
    divergence_pct: float
    preceding_context: str | None = None


@dataclass
class RetryResult:
    """Result of a single retry attempt"""

    node_id: str
    strategy: str  # "continuation" or "fresh"
    trial: int
    target_tokens: int
    actual_tokens: int
    deviation_pct: float
    success: bool  # Within 10% of target
    retries_used: int

    def to_dict(self):
        return {
            "node_id": self.node_id,
            "strategy": self.strategy,
            "trial": self.trial,
            "target_tokens": self.target_tokens,
            "actual_tokens": self.actual_tokens,
            "deviation_pct": self.deviation_pct,
            "success": self.success,
            "retries_used": self.retries_used,
        }


class RetryStrategyTester:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        self.semaphore = asyncio.Semaphore(20)  # Limit concurrent API calls

        # Check what columns the database has
        cursor = self.conn.execute("PRAGMA table_info(tree_nodes)")
        columns = [row[1] for row in cursor.fetchall()]
        self.has_token_count = "token_count" in columns

        # Determine target token count (similar to analyze-outlier-nodes.py)
        if not self.has_token_count:
            self.target_tokens = 200  # Default
        else:
            cursor = self.conn.execute(
                "SELECT token_count FROM tree_nodes WHERE left_child_id IS NULL AND token_count IS NOT NULL LIMIT 100"
            )
            leaf_tokens = [row["token_count"] for row in cursor.fetchall()]
            if leaf_tokens:
                self.target_tokens = int(statistics.median(leaf_tokens))
            else:
                self.target_tokens = 200

        print(f"Target token count: {self.target_tokens}")

    def _get_node(self, node_id: str) -> sqlite3.Row | None:
        """Get node from database."""
        if not node_id:
            return None
        cursor = self.conn.execute("SELECT * FROM tree_nodes WHERE id = ?", (node_id,))
        return cursor.fetchone()

    def get_outlier_nodes(self, limit: int = 20) -> list[SummaryCase]:
        """Get outlier nodes from database with >20% deviation"""
        if self.has_token_count:
            query = """
            SELECT
                n.id as node_id,
                n.text as summary,
                n.token_count as original_tokens,
                n.left_child_id,
                n.right_child_id,
                n.preceding_neighbor_id,
                ABS(n.token_count - ?) * 100.0 / ? as divergence_pct
            FROM tree_nodes n
            WHERE n.left_child_id IS NOT NULL  -- Non-leaf nodes only
              AND n.token_count IS NOT NULL
              AND ABS(n.token_count - ?) * 100.0 / ? > 20  -- Only outliers
            ORDER BY divergence_pct DESC
            LIMIT ?
            """
            cursor = self.conn.execute(
                query,
                (
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    limit,
                ),
            )
        else:
            # Use text length approximation
            query = """
            SELECT
                n.id as node_id,
                n.text as summary,
                LENGTH(n.text) / 4 as original_tokens,
                n.left_child_id,
                n.right_child_id,
                n.preceding_neighbor_id,
                ABS(LENGTH(n.text) / 4 - ?) * 100.0 / ? as divergence_pct
            FROM tree_nodes n
            WHERE n.left_child_id IS NOT NULL
              AND n.text IS NOT NULL
              AND ABS(LENGTH(n.text) / 4 - ?) * 100.0 / ? > 20
            ORDER BY divergence_pct DESC
            LIMIT ?
            """
            cursor = self.conn.execute(
                query,
                (
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    self.target_tokens,
                    limit,
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

            cases.append(
                SummaryCase(
                    node_id=row["node_id"],
                    left_text=left_child["text"],
                    right_text=right_child["text"],
                    original_summary=row["summary"],
                    original_tokens=int(row["original_tokens"]),
                    target_tokens=self.target_tokens,
                    divergence_pct=row["divergence_pct"],
                    preceding_context=preceding_context,
                )
            )

        return cases

    def count_tokens(self, text: str) -> int:
        """Count tokens in text"""
        return len(self.encoding.encode(text))

    def build_prompt_messages(self, case: SummaryCase) -> tuple[str, str]:
        """Build the system and user messages for a case"""
        target_words = int(case.target_tokens * WORDS_PER_TOKEN)

        # Build the combined text to summarize
        text_to_summarize = f"{case.left_text}\n\n{case.right_text}"

        # System message (simplified version from ragzoom/index.py)
        system_message = "You are a precise text summarizer."

        # User message (based on ragzoom/index.py format)
        instruction = f"""You will be given a piece of content to summarize, embedded within the context in which it appears in a source document. You are to summarize only the content between the <SUMMARIZE_TEXT> tags in as close to {target_words} words as possible. There is some tolerance, but I need you to get as close as possible to the target word count. Do not include information from the context."""

        if case.preceding_context:
            user_message = f"""{instruction}

<CONTEXT>
{case.preceding_context}
</CONTEXT>

<SUMMARIZE_TEXT>
{text_to_summarize}
</SUMMARIZE_TEXT>

Write your summary."""
        else:
            user_message = f"""{instruction}

<SUMMARIZE_TEXT>
{text_to_summarize}
</SUMMARIZE_TEXT>

Write your summary."""

        return system_message, user_message

    async def test_continuation_strategy(
        self, case: SummaryCase, trial: int
    ) -> RetryResult:
        """Test the continuation strategy (current approach)"""
        async with self.semaphore:
            system_msg, user_msg = self.build_prompt_messages(case)
            target_tokens = case.target_tokens
            target_words = int(target_tokens * WORDS_PER_TOKEN)

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            # First attempt
            response = await self.client.chat.completions.create(
                model=MODEL, messages=messages, reasoning_effort="minimal"
            )

            first_summary = response.choices[0].message.content
            first_tokens = self.count_tokens(first_summary)
            deviation_pct = 100.0 * (first_tokens - target_tokens) / target_tokens

            # Check if retry needed
            if abs(deviation_pct) <= 10:
                return RetryResult(
                    node_id=case.node_id,
                    strategy="continuation",
                    trial=trial,
                    target_tokens=target_tokens,
                    actual_tokens=first_tokens,
                    deviation_pct=deviation_pct,
                    success=True,
                    retries_used=0,
                )

            # Need retry - continue conversation
            direction = "larger" if deviation_pct > 0 else "smaller"
            retry_prompt = f"Your summary was {abs(deviation_pct):.0f}% {direction} than the target length. Think carefully about how to write it in as close to {target_words} words as possible and try again."

            messages.append({"role": "assistant", "content": first_summary})
            messages.append({"role": "user", "content": retry_prompt})

            retry_response = await self.client.chat.completions.create(
                model=MODEL, messages=messages, reasoning_effort="minimal"
            )

            retry_summary = retry_response.choices[0].message.content
            retry_tokens = self.count_tokens(retry_summary)
            retry_deviation = 100.0 * (retry_tokens - target_tokens) / target_tokens

            return RetryResult(
                node_id=case.node_id,
                strategy="continuation",
                trial=trial,
                target_tokens=target_tokens,
                actual_tokens=retry_tokens,
                deviation_pct=retry_deviation,
                success=abs(retry_deviation) <= 10,
                retries_used=1,
            )

    async def test_fresh_strategy(self, case: SummaryCase, trial: int) -> RetryResult:
        """Test the fresh restart strategy"""
        async with self.semaphore:
            system_msg, user_msg = self.build_prompt_messages(case)
            target_tokens = case.target_tokens

            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]

            # First attempt
            response = await self.client.chat.completions.create(
                model=MODEL, messages=messages, reasoning_effort="minimal"
            )

            first_summary = response.choices[0].message.content
            first_tokens = self.count_tokens(first_summary)
            deviation_pct = 100.0 * (first_tokens - target_tokens) / target_tokens

            # Check if retry needed
            if abs(deviation_pct) <= 10:
                return RetryResult(
                    node_id=case.node_id,
                    strategy="fresh",
                    trial=trial,
                    target_tokens=target_tokens,
                    actual_tokens=first_tokens,
                    deviation_pct=deviation_pct,
                    success=True,
                    retries_used=0,
                )

            # Need retry - start fresh conversation with same prompt
            retry_response = await self.client.chat.completions.create(
                model=MODEL,
                messages=messages,  # Same original messages
                reasoning_effort="minimal",
            )

            retry_summary = retry_response.choices[0].message.content
            retry_tokens = self.count_tokens(retry_summary)
            retry_deviation = 100.0 * (retry_tokens - target_tokens) / target_tokens

            return RetryResult(
                node_id=case.node_id,
                strategy="fresh",
                trial=trial,
                target_tokens=target_tokens,
                actual_tokens=retry_tokens,
                deviation_pct=retry_deviation,
                success=abs(retry_deviation) <= 10,
                retries_used=1,
            )

    async def run_trials(self, case: SummaryCase, trials_per_strategy: int = 3):
        """Run multiple trials for both strategies on a single case"""
        # Create all trial tasks
        cont_tasks = [
            self.test_continuation_strategy(case, i + 1)
            for i in range(trials_per_strategy)
        ]
        fresh_tasks = [
            self.test_fresh_strategy(case, i + 1) for i in range(trials_per_strategy)
        ]

        # Run all trials in parallel
        all_tasks = cont_tasks + fresh_tasks
        results = await asyncio.gather(*all_tasks)

        # Print results
        cont_results = results[:trials_per_strategy]
        fresh_results = results[trials_per_strategy:]

        for i, result in enumerate(cont_results):
            print(
                f"  Continuation trial {i + 1}: deviation={result.deviation_pct:.1f}%, success={result.success}"
            )
        for i, result in enumerate(fresh_results):
            print(
                f"  Fresh trial {i + 1}: deviation={result.deviation_pct:.1f}%, success={result.success}"
            )

        return results

    async def run_experiment(self, max_nodes: int = 10, trials_per_strategy: int = 3):
        """Run the full experiment"""
        print(f"Loading outlier nodes from {self.db_path}...")
        cases = self.get_outlier_nodes(max_nodes)

        if not cases:
            print("No outlier nodes found in database")
            return

        print(f"Found {len(cases)} outlier cases")
        print(f"Running {trials_per_strategy} trials per strategy for each case\n")

        all_results = []

        for i, case in enumerate(cases, 1):
            print(
                f"Case {i}/{len(cases)}: {case.node_id} (original deviation: {case.divergence_pct:.1f}%)"
            )
            results = await self.run_trials(case, trials_per_strategy)
            all_results.extend(results)
            print()

        # Analyze results
        self.analyze_results(all_results)

        # Save raw results
        output_file = Path(
            f"retry_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_file, "w") as f:
            json.dump([r.to_dict() for r in all_results], f, indent=2)
        print(f"\nRaw results saved to {output_file}")

    def analyze_results(self, results: list[RetryResult]):
        """Analyze and print experiment results"""
        continuation_results = [r for r in results if r.strategy == "continuation"]
        fresh_results = [r for r in results if r.strategy == "fresh"]

        print("\n" + "=" * 60)
        print("EXPERIMENT RESULTS")
        print("=" * 60)

        # Success rates
        cont_success = (
            sum(1 for r in continuation_results if r.success)
            / len(continuation_results)
            * 100
        )
        fresh_success = (
            sum(1 for r in fresh_results if r.success) / len(fresh_results) * 100
        )

        print("\nSuccess Rate (within 10% of target):")
        print(
            f"  Continuation: {cont_success:.1f}% ({sum(1 for r in continuation_results if r.success)}/{len(continuation_results)})"
        )
        print(
            f"  Fresh:        {fresh_success:.1f}% ({sum(1 for r in fresh_results if r.success)}/{len(fresh_results)})"
        )

        # Average deviation
        cont_avg_dev = sum(abs(r.deviation_pct) for r in continuation_results) / len(
            continuation_results
        )
        fresh_avg_dev = sum(abs(r.deviation_pct) for r in fresh_results) / len(
            fresh_results
        )

        print("\nAverage Absolute Deviation:")
        print(f"  Continuation: {cont_avg_dev:.1f}%")
        print(f"  Fresh:        {fresh_avg_dev:.1f}%")

        # Deviation distribution
        def get_distribution(results):
            buckets = defaultdict(int)
            for r in results:
                if abs(r.deviation_pct) <= 10:
                    buckets["0-10%"] += 1
                elif abs(r.deviation_pct) <= 20:
                    buckets["10-20%"] += 1
                elif abs(r.deviation_pct) <= 30:
                    buckets["20-30%"] += 1
                else:
                    buckets[">30%"] += 1
            return buckets

        print("\nDeviation Distribution:")
        cont_dist = get_distribution(continuation_results)
        fresh_dist = get_distribution(fresh_results)

        for bucket in ["0-10%", "10-20%", "20-30%", ">30%"]:
            print(
                f"  {bucket:8} - Continuation: {cont_dist[bucket]:3d}, Fresh: {fresh_dist[bucket]:3d}"
            )

        # Per-node comparison
        print("\nPer-Node Performance (average across trials):")
        node_ids = list(set(r.node_id for r in results))

        better_with_continuation = 0
        better_with_fresh = 0

        for node_id in node_ids[:5]:  # Show first 5 for brevity
            cont_node = [r for r in continuation_results if r.node_id == node_id]
            fresh_node = [r for r in fresh_results if r.node_id == node_id]

            cont_avg = sum(abs(r.deviation_pct) for r in cont_node) / len(cont_node)
            fresh_avg = sum(abs(r.deviation_pct) for r in fresh_node) / len(fresh_node)

            winner = (
                "CONT"
                if cont_avg < fresh_avg
                else "FRESH" if fresh_avg < cont_avg else "TIE"
            )
            if cont_avg < fresh_avg:
                better_with_continuation += 1
            elif fresh_avg < cont_avg:
                better_with_fresh += 1

            print(
                f"  {node_id[:8]}... Cont: {cont_avg:5.1f}%, Fresh: {fresh_avg:5.1f}% [{winner}]"
            )

        if len(node_ids) > 5:
            print(f"  ... and {len(node_ids) - 5} more nodes")

        # Overall winner count
        for node_id in node_ids:
            cont_node = [r for r in continuation_results if r.node_id == node_id]
            fresh_node = [r for r in fresh_results if r.node_id == node_id]

            cont_avg = sum(abs(r.deviation_pct) for r in cont_node) / len(cont_node)
            fresh_avg = sum(abs(r.deviation_pct) for r in fresh_node) / len(fresh_node)

            if cont_avg < fresh_avg:
                better_with_continuation += 1
            elif fresh_avg < cont_avg:
                better_with_fresh += 1

        print("\nNodes performing better with:")
        print(f"  Continuation: {better_with_continuation}")
        print(f"  Fresh:        {better_with_fresh}")
        print(
            f"  Tie:          {len(node_ids) - better_with_continuation - better_with_fresh}"
        )


@click.command()
@click.option(
    "--database",
    type=click.Path(exists=True),
    default="benchmarks_words/ragzoom.db",
    help="Path to database",
)
@click.option(
    "--max-nodes", type=int, default=10, help="Maximum number of nodes to test"
)
@click.option(
    "--trials", type=int, default=3, help="Number of trials per strategy per node"
)
def main(database, max_nodes, trials):
    """Test retry strategies for summarization"""
    db_path = Path(database)

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return

    tester = RetryStrategyTester(db_path)
    asyncio.run(tester.run_experiment(max_nodes, trials))


if __name__ == "__main__":
    main()
