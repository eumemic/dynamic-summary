#!/usr/bin/env python3
"""Interactive telemetry explorer for deep analysis of RagZoom benchmarks.

This tool provides an interactive CLI for exploring telemetry data, finding
outliers, and performing temporal analysis on indexing operations.

Usage:
    python scripts/telemetry_explorer.py benchmark_results/metrics_200_tokens.json
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for importing ragzoom
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import pandas as pd
    from rich import box
    from rich.console import Console
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.tree import Tree
except ImportError as e:
    print(f"Error: Missing required dependencies: {e}")
    print("Please install: pip install pandas rich")
    sys.exit(1)

from ragzoom.telemetry import parse_telemetry_format

console = Console()

# Constants for outlier detection
HIGH_INPUT_AMPLIFICATION_THRESHOLD = 3.0
MULTIPLE_RETRY_THRESHOLD = 1

class TelemetryExplorer:
    """Interactive explorer for telemetry data."""

    def __init__(self, telemetry_data: dict):
        """Initialize explorer with telemetry data."""
        self.telemetry = parse_telemetry_format(telemetry_data)
        self.nodes_df = self._create_nodes_dataframe()

    def _create_nodes_dataframe(self) -> pd.DataFrame:
        """Convert telemetry nodes to pandas DataFrame for easy analysis."""
        rows = []

        for doc_type, doc_data in self.telemetry["documents"].items():
            for node in doc_data.get("nodes", []):
                row = {
                    "document": doc_type,
                    "node_id": node.get("node_id"),
                    "node_type": node.get("node_type"),
                    "level": node.get("level"),
                    "span_start": node.get("span", [0, 0])[0],
                    "span_end": node.get("span", [0, 0])[1],
                    "span_size": node.get("span", [0, 0])[1] - node.get("span", [0, 0])[0],
                    "created_at": node.get("created_at", 0),
                }

                # Add embedding info
                if "embedding" in node and node["embedding"]:
                    emb = node["embedding"]
                    row.update({
                        "embedding_tokens": emb.get("text_tokens", 0),
                        "batch_size": emb.get("batch_size", 0),
                        "batch_position": emb.get("batch_position", 0),
                        "embedding_model": emb.get("model", ""),
                        "embedding_timestamp": emb.get("timestamp", 0),
                    })

                # Add summary info
                if "summary_attempts" in node and node["summary_attempts"]:
                    # Use the accepted attempt if available
                    accepted = None
                    for attempt in node["summary_attempts"]:
                        if attempt.get("status") == "accepted":
                            accepted = attempt
                            break

                    if accepted:
                        row.update({
                            "target_tokens": accepted.get("target_tokens", 0),
                            "actual_tokens": accepted.get("actual_tokens", 0),
                            "prompt_tokens": accepted.get("prompt_tokens", 0),
                            "completion_tokens": accepted.get("completion_tokens", 0),
                            "input_text_tokens": accepted.get("input_text_tokens", 0),
                            "summary_model": accepted.get("model", ""),
                            "retry_count": len(node["summary_attempts"]) - 1,
                        })

                        # Calculate amplifications
                        if accepted.get("input_text_tokens", 0) > 0:
                            row["input_amplification"] = (
                                accepted.get("prompt_tokens", 0) /
                                accepted.get("input_text_tokens", 1)
                            )

                        if accepted.get("actual_tokens", 0) > 0:
                            row["output_amplification"] = (
                                accepted.get("completion_tokens", 0) /
                                accepted.get("actual_tokens", 1)
                            )

                rows.append(row)

        return pd.DataFrame(rows)

    def run_interactive_session(self) -> None:
        """Run the interactive exploration session."""
        console.print("\n[bold cyan]RagZoom Telemetry Explorer[/bold cyan]")
        console.print(f"Loaded {len(self.nodes_df)} nodes from telemetry data\n")

        while True:
            self._show_menu()
            choice = Prompt.ask(
                "\nSelect an option",
                choices=["1", "2", "3", "4", "5", "6", "7", "8", "q"],
                default="1"
            )

            if choice == "q":
                console.print("\n[yellow]Exiting explorer...[/yellow]")
                break
            elif choice == "1":
                self._browse_nodes()
            elif choice == "2":
                self._show_node_details()
            elif choice == "3":
                self._temporal_analysis()
            elif choice == "4":
                self._find_outliers()
            elif choice == "5":
                self._show_statistics()
            elif choice == "6":
                self._filter_nodes()
            elif choice == "7":
                self._export_data()
            elif choice == "8":
                self._batch_analysis()

    def _show_menu(self) -> None:
        """Display main menu."""
        table = Table(title="Main Menu", box=box.ROUNDED)
        table.add_column("Option", style="cyan", width=10)
        table.add_column("Description", style="green")

        table.add_row("1", "Browse nodes")
        table.add_row("2", "Show node details")
        table.add_row("3", "Temporal analysis")
        table.add_row("4", "Find outliers")
        table.add_row("5", "Show statistics")
        table.add_row("6", "Filter nodes")
        table.add_row("7", "Export data")
        table.add_row("8", "Batch analysis")
        table.add_row("q", "Quit")

        console.print(table)

    def _browse_nodes(self) -> None:
        """Browse nodes with pagination."""
        page_size = 20
        page = 0

        # Ask for filtering
        filter_type = Prompt.ask(
            "Filter by node type?",
            choices=["all", "leaf", "summary"],
            default="all"
        )

        if filter_type == "all":
            df = self.nodes_df
        else:
            df = self.nodes_df[self.nodes_df["node_type"] == filter_type]

        total_pages = (len(df) + page_size - 1) // page_size

        while True:
            start_idx = page * page_size
            end_idx = min((page + 1) * page_size, len(df))

            table = Table(
                title=f"Nodes (Page {page + 1}/{total_pages})",
                box=box.ROUNDED
            )
            table.add_column("Index", style="dim", width=6)
            table.add_column("Node ID", style="cyan")
            table.add_column("Type", style="green", width=8)
            table.add_column("Level", style="yellow", width=6)
            table.add_column("Span Size", style="blue", width=10)
            table.add_column("Tokens", style="magenta", width=8)

            for idx in range(start_idx, end_idx):
                row = df.iloc[idx]
                tokens = row.get("embedding_tokens") or row.get("actual_tokens", "")
                table.add_row(
                    str(idx),
                    row["node_id"][:20] + "..." if len(row["node_id"]) > 20 else row["node_id"],
                    row["node_type"],
                    str(row["level"]),
                    str(row["span_size"]),
                    str(tokens) if tokens else "-"
                )

            console.print(table)

            action = Prompt.ask(
                "\n[n]ext, [p]revious, [d]etails, [q]uit",
                choices=["n", "p", "d", "q"],
                default="n"
            )

            if action == "q":
                break
            elif action == "n" and page < total_pages - 1:
                page += 1
            elif action == "p" and page > 0:
                page -= 1
            elif action == "d":
                idx_str = Prompt.ask("Enter node index", default=str(start_idx))
                try:
                    idx = int(idx_str)
                    self._show_specific_node_details(df.iloc[idx])
                except (ValueError, IndexError):
                    console.print("[red]Invalid index[/red]")

    def _show_node_details(self) -> None:
        """Show details for a specific node."""
        node_id = Prompt.ask("Enter node ID (or partial ID)")

        # Find matching nodes
        matches = self.nodes_df[
            self.nodes_df["node_id"].str.contains(node_id, case=False)
        ]

        if len(matches) == 0:
            console.print(f"[red]No nodes found matching '{node_id}'[/red]")
        elif len(matches) == 1:
            self._show_specific_node_details(matches.iloc[0])
        else:
            console.print(f"[yellow]Found {len(matches)} matches:[/yellow]")
            for idx, row in matches.iterrows():
                console.print(f"  - {row['node_id']} ({row['node_type']}, level {row['level']})")

            specific_id = Prompt.ask("Enter full node ID")
            specific = self.nodes_df[self.nodes_df["node_id"] == specific_id]
            if len(specific) == 1:
                self._show_specific_node_details(specific.iloc[0])

    def _show_specific_node_details(self, node: pd.Series) -> None:
        """Display detailed information for a specific node."""
        # Find the full node data from telemetry
        full_node = None
        for doc_data in self.telemetry["documents"].values():
            for n in doc_data.get("nodes", []):
                if n.get("node_id") == node["node_id"]:
                    full_node = n
                    break

        if not full_node:
            console.print("[red]Could not find full node data[/red]")
            return

        # Create detail tree
        tree = Tree(f"[bold cyan]Node: {node['node_id']}[/bold cyan]")

        # Basic info
        basic = tree.add("[yellow]Basic Information[/yellow]")
        basic.add(f"Type: {node['node_type']}")
        basic.add(f"Level: {node['level']}")
        basic.add(f"Span: [{node['span_start']}:{node['span_end']}] ({node['span_size']} chars)")
        basic.add(f"Created at: {node['created_at']:.2f}")

        # Embedding info
        if "embedding" in full_node and full_node["embedding"]:
            emb = full_node["embedding"]
            emb_tree = tree.add("[green]Embedding[/green]")
            emb_tree.add(f"Tokens: {emb['text_tokens']}")
            emb_tree.add(f"Model: {emb['model']}")
            emb_tree.add(f"Batch: {emb['batch_position'] + 1}/{emb['batch_size']}")
            emb_tree.add(f"Timestamp: {emb['timestamp']:.2f}")

        # Summary attempts
        if "summary_attempts" in full_node and full_node["summary_attempts"]:
            summary_tree = tree.add("[magenta]Summary Attempts[/magenta]")

            for i, attempt in enumerate(full_node["summary_attempts"]):
                attempt_tree = summary_tree.add(
                    f"Attempt {i + 1} ({'Retry' if attempt['is_retry'] else 'Initial'})"
                )
                attempt_tree.add(f"Status: {attempt['status']}")
                attempt_tree.add(f"Target: {attempt['target_tokens']} tokens")
                attempt_tree.add(f"Actual: {attempt['actual_tokens']} tokens")
                attempt_tree.add(f"Prompt: {attempt['prompt_tokens']} tokens")
                attempt_tree.add(f"Completion: {attempt['completion_tokens']} tokens")
                attempt_tree.add(f"Model: {attempt['model']}")

                if attempt.get('rejection_reason'):
                    attempt_tree.add(f"[red]Rejection: {attempt['rejection_reason']}[/red]")

                # Show amplification for accepted attempts
                if attempt['status'] == 'accepted' and attempt['input_text_tokens'] > 0:
                    input_amp = attempt['prompt_tokens'] / attempt['input_text_tokens']
                    output_amp = attempt['completion_tokens'] / attempt['actual_tokens'] if attempt['actual_tokens'] > 0 else 1.0
                    attempt_tree.add(f"[yellow]Input amplification: {input_amp:.2f}x[/yellow]")
                    attempt_tree.add(f"[yellow]Output amplification: {output_amp:.2f}x[/yellow]")

        console.print(tree)

        # Wait for user
        Prompt.ask("\nPress Enter to continue")

    def _temporal_analysis(self) -> None:
        """Analyze temporal patterns in node creation."""
        # Sort by creation time
        df = self.nodes_df.sort_values("created_at")

        if len(df) == 0:
            console.print("[red]No nodes with timestamp data[/red]")
            return

        # Calculate time range
        start_time = df["created_at"].min()
        end_time = df["created_at"].max()
        duration = end_time - start_time

        console.print("\n[bold]Temporal Analysis[/bold]")
        console.print(f"Total duration: {duration:.2f} seconds")
        console.print(f"Start time: {start_time:.2f}")
        console.print(f"End time: {end_time:.2f}")

        # Show creation rate by level
        table = Table(title="Node Creation by Level", box=box.ROUNDED)
        table.add_column("Level", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("First Created", style="yellow")
        table.add_column("Last Created", style="yellow")
        table.add_column("Duration (s)", style="magenta")

        for level in sorted(df["level"].unique()):
            level_df = df[df["level"] == level]
            first = level_df["created_at"].min()
            last = level_df["created_at"].max()

            table.add_row(
                str(level),
                str(len(level_df)),
                f"{first - start_time:.2f}",
                f"{last - start_time:.2f}",
                f"{last - first:.2f}"
            )

        console.print(table)

        # Find slow operations
        console.print("\n[bold]Slowest Operations[/bold]")

        # Calculate operation durations (time between consecutive nodes)
        df = df.copy()
        df["duration"] = df["created_at"].diff()

        # Show top 10 slowest
        slow_ops = df.nlargest(10, "duration", keep="all")

        table = Table(title="Top 10 Slowest Operations", box=box.ROUNDED)
        table.add_column("Node ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Level", style="yellow")
        table.add_column("Duration (s)", style="red")

        for _, row in slow_ops.iterrows():
            if pd.notna(row["duration"]):
                table.add_row(
                    row["node_id"][:30] + "..." if len(row["node_id"]) > 30 else row["node_id"],
                    row["node_type"],
                    str(row["level"]),
                    f"{row['duration']:.3f}"
                )

        console.print(table)

    def _find_outliers(self) -> None:
        """Find nodes with unusual characteristics."""
        console.print("\n[bold]Outlier Detection[/bold]\n")

        # High amplification outliers
        if "input_amplification" in self.nodes_df.columns:
            high_amp = self.nodes_df[self.nodes_df["input_amplification"] > HIGH_INPUT_AMPLIFICATION_THRESHOLD]
            if len(high_amp) > 0:
                console.print(f"[yellow]Found {len(high_amp)} nodes with input amplification > {HIGH_INPUT_AMPLIFICATION_THRESHOLD}x:[/yellow]")
                for _, row in high_amp.head(5).iterrows():
                    console.print(
                        f"  - {row['node_id'][:40]}... "
                        f"({row['input_amplification']:.2f}x amplification)"
                    )
                if len(high_amp) > 5:
                    console.print(f"  ... and {len(high_amp) - 5} more")
            else:
                console.print("[green]No nodes with high input amplification[/green]")

        # High retry count
        if "retry_count" in self.nodes_df.columns:
            high_retry = self.nodes_df[self.nodes_df["retry_count"] > MULTIPLE_RETRY_THRESHOLD]
            if len(high_retry) > 0:
                console.print(f"\n[yellow]Found {len(high_retry)} nodes with multiple retries:[/yellow]")
                for _, row in high_retry.head(5).iterrows():
                    console.print(
                        f"  - {row['node_id'][:40]}... "
                        f"({row['retry_count']} retries)"
                    )
            else:
                console.print("\n[green]No nodes with multiple retries[/green]")

        # Token count outliers
        if "embedding_tokens" in self.nodes_df.columns:
            emb_df = self.nodes_df[self.nodes_df["embedding_tokens"].notna()]
            if len(emb_df) > 0:
                q1 = emb_df["embedding_tokens"].quantile(0.25)
                q3 = emb_df["embedding_tokens"].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                outliers = emb_df[
                    (emb_df["embedding_tokens"] < lower_bound) |
                    (emb_df["embedding_tokens"] > upper_bound)
                ]

                if len(outliers) > 0:
                    console.print(
                        f"\n[yellow]Found {len(outliers)} nodes with outlier token counts:[/yellow]"
                    )
                    console.print(f"  Normal range: {lower_bound:.0f} - {upper_bound:.0f} tokens")
                    for _, row in outliers.head(5).iterrows():
                        console.print(
                            f"  - {row['node_id'][:40]}... "
                            f"({row['embedding_tokens']:.0f} tokens)"
                        )
                else:
                    console.print("\n[green]No token count outliers[/green]")

    def _show_statistics(self) -> None:
        """Display overall statistics."""
        console.print("\n[bold]Telemetry Statistics[/bold]\n")

        # Basic counts
        table = Table(title="Node Counts", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total nodes", str(len(self.nodes_df)))
        table.add_row(
            "Leaf nodes",
            str(len(self.nodes_df[self.nodes_df["node_type"] == "leaf"]))
        )
        table.add_row(
            "Summary nodes",
            str(len(self.nodes_df[self.nodes_df["node_type"] == "summary"]))
        )
        table.add_row(
            "Unique documents",
            str(self.nodes_df["document"].nunique())
        )
        table.add_row(
            "Max tree level",
            str(self.nodes_df["level"].max())
        )

        console.print(table)

        # Token statistics
        if "embedding_tokens" in self.nodes_df.columns:
            emb_df = self.nodes_df[self.nodes_df["embedding_tokens"].notna()]
            if len(emb_df) > 0:
                table = Table(title="Embedding Token Statistics", box=box.ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Mean", f"{emb_df['embedding_tokens'].mean():.1f}")
                table.add_row("Median", f"{emb_df['embedding_tokens'].median():.1f}")
                table.add_row("Std Dev", f"{emb_df['embedding_tokens'].std():.1f}")
                table.add_row("Min", f"{emb_df['embedding_tokens'].min():.0f}")
                table.add_row("Max", f"{emb_df['embedding_tokens'].max():.0f}")

                console.print(table)

        # Summary statistics
        if "actual_tokens" in self.nodes_df.columns:
            sum_df = self.nodes_df[self.nodes_df["actual_tokens"].notna()]
            if len(sum_df) > 0:
                table = Table(title="Summary Token Statistics", box=box.ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                table.add_row("Mean", f"{sum_df['actual_tokens'].mean():.1f}")
                table.add_row("Median", f"{sum_df['actual_tokens'].median():.1f}")
                table.add_row("Std Dev", f"{sum_df['actual_tokens'].std():.1f}")
                table.add_row("Min", f"{sum_df['actual_tokens'].min():.0f}")
                table.add_row("Max", f"{sum_df['actual_tokens'].max():.0f}")

                console.print(table)

        # Amplification statistics
        if "input_amplification" in self.nodes_df.columns:
            amp_df = self.nodes_df[self.nodes_df["input_amplification"].notna()]
            if len(amp_df) > 0:
                table = Table(title="Amplification Statistics", box=box.ROUNDED)
                table.add_column("Metric", style="cyan")
                table.add_column("Input Amp", style="yellow")
                table.add_column("Output Amp", style="magenta")

                table.add_row(
                    "Mean",
                    f"{amp_df['input_amplification'].mean():.2f}x",
                    f"{amp_df['output_amplification'].mean():.2f}x" if 'output_amplification' in amp_df.columns else "-"
                )
                table.add_row(
                    "Median",
                    f"{amp_df['input_amplification'].median():.2f}x",
                    f"{amp_df['output_amplification'].median():.2f}x" if 'output_amplification' in amp_df.columns else "-"
                )
                table.add_row(
                    "P90",
                    f"{amp_df['input_amplification'].quantile(0.9):.2f}x",
                    f"{amp_df['output_amplification'].quantile(0.9):.2f}x" if 'output_amplification' in amp_df.columns else "-"
                )

                console.print(table)

    def _filter_nodes(self) -> None:
        """Apply filters to node data."""
        console.print("\n[bold]Filter Nodes[/bold]")

        # Start with all nodes
        filtered = self.nodes_df.copy()

        # Node type filter
        node_type = Prompt.ask(
            "Filter by node type",
            choices=["all", "leaf", "summary"],
            default="all"
        )
        if node_type != "all":
            filtered = filtered[filtered["node_type"] == node_type]

        # Level filter
        if Confirm.ask("Filter by level?", default=False):
            level = int(Prompt.ask("Enter level"))
            filtered = filtered[filtered["level"] == level]

        # Token range filter
        if "embedding_tokens" in filtered.columns or "actual_tokens" in filtered.columns:
            if Confirm.ask("Filter by token count?", default=False):
                min_tokens = int(Prompt.ask("Minimum tokens", default="0"))
                max_tokens = int(Prompt.ask("Maximum tokens", default="1000"))

                if "embedding_tokens" in filtered.columns:
                    filtered = filtered[
                        (filtered["embedding_tokens"] >= min_tokens) &
                        (filtered["embedding_tokens"] <= max_tokens)
                    ]
                elif "actual_tokens" in filtered.columns:
                    filtered = filtered[
                        (filtered["actual_tokens"] >= min_tokens) &
                        (filtered["actual_tokens"] <= max_tokens)
                    ]

        # Amplification filter
        if "input_amplification" in filtered.columns:
            if Confirm.ask("Filter by amplification?", default=False):
                min_amp = float(Prompt.ask("Minimum amplification", default="1.0"))
                filtered = filtered[filtered["input_amplification"] >= min_amp]

        console.print(f"\n[green]Filtered to {len(filtered)} nodes[/green]")

        # Show filtered results
        if len(filtered) > 0:
            table = Table(title="Filtered Nodes (Top 10)", box=box.ROUNDED)
            table.add_column("Node ID", style="cyan")
            table.add_column("Type", style="green")
            table.add_column("Level", style="yellow")
            table.add_column("Tokens", style="magenta")

            for _, row in filtered.head(10).iterrows():
                tokens = row.get("embedding_tokens") or row.get("actual_tokens", "")
                table.add_row(
                    row["node_id"][:30] + "..." if len(row["node_id"]) > 30 else row["node_id"],
                    row["node_type"],
                    str(row["level"]),
                    str(tokens) if tokens else "-"
                )

            console.print(table)

            if len(filtered) > 10:
                console.print(f"\n... and {len(filtered) - 10} more nodes")

            # Export option
            if Confirm.ask("\nExport filtered results?", default=False):
                self._export_dataframe(filtered, "filtered_nodes")

    def _export_data(self) -> None:
        """Export telemetry data."""
        console.print("\n[bold]Export Options[/bold]")

        export_type = Prompt.ask(
            "What to export?",
            choices=["all", "nodes", "summary", "custom"],
            default="all"
        )

        if export_type == "all":
            self._export_dataframe(self.nodes_df, "all_nodes")
        elif export_type == "nodes":
            node_type = Prompt.ask(
                "Node type",
                choices=["all", "leaf", "summary"],
                default="all"
            )
            if node_type == "all":
                df = self.nodes_df
            else:
                df = self.nodes_df[self.nodes_df["node_type"] == node_type]
            self._export_dataframe(df, f"{node_type}_nodes")
        elif export_type == "summary":
            self._export_summary_report()
        elif export_type == "custom":
            console.print("[yellow]Use the filter option first, then export[/yellow]")

    def _export_dataframe(self, df: pd.DataFrame, prefix: str) -> None:
        """Export a dataframe to CSV or JSON."""
        format_type = Prompt.ask(
            "Export format",
            choices=["csv", "json"],
            default="csv"
        )

        filename = f"{prefix}_export.{format_type}"

        if format_type == "csv":
            df.to_csv(filename, index=False)
        else:
            df.to_json(filename, orient="records", indent=2)

        console.print(f"[green]Exported {len(df)} rows to {filename}[/green]")

    def _export_summary_report(self) -> None:
        """Export a summary report."""
        report = {
            "total_nodes": len(self.nodes_df),
            "leaf_nodes": len(self.nodes_df[self.nodes_df["node_type"] == "leaf"]),
            "summary_nodes": len(self.nodes_df[self.nodes_df["node_type"] == "summary"]),
            "max_level": int(self.nodes_df["level"].max()),
            "documents": list(self.nodes_df["document"].unique()),
        }

        # Add token statistics
        if "embedding_tokens" in self.nodes_df.columns:
            emb_stats = self.nodes_df["embedding_tokens"].describe()
            report["embedding_stats"] = emb_stats.to_dict()

        if "actual_tokens" in self.nodes_df.columns:
            sum_stats = self.nodes_df["actual_tokens"].describe()
            report["summary_stats"] = sum_stats.to_dict()

        # Add amplification stats
        if "input_amplification" in self.nodes_df.columns:
            amp_df = self.nodes_df[self.nodes_df["input_amplification"].notna()]
            if len(amp_df) > 0:
                report["amplification"] = {
                    "mean_input": float(amp_df["input_amplification"].mean()),
                    "median_input": float(amp_df["input_amplification"].median()),
                    "p90_input": float(amp_df["input_amplification"].quantile(0.9)),
                    "p95_input": float(amp_df["input_amplification"].quantile(0.95)),
                }

        filename = "telemetry_summary_report.json"
        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        console.print(f"[green]Exported summary report to {filename}[/green]")

    def _batch_analysis(self) -> None:
        """Analyze embedding batch patterns."""
        if "batch_size" not in self.nodes_df.columns:
            console.print("[red]No batch data available[/red]")
            return

        console.print("\n[bold]Batch Analysis[/bold]\n")

        # Get nodes with batch info
        batch_df = self.nodes_df[self.nodes_df["batch_size"].notna()].copy()

        if len(batch_df) == 0:
            console.print("[red]No nodes with batch information[/red]")
            return

        # Batch size distribution
        table = Table(title="Batch Size Distribution", box=box.ROUNDED)
        table.add_column("Batch Size", style="cyan")
        table.add_column("Count", style="green")
        table.add_column("Percentage", style="yellow")

        size_counts = batch_df["batch_size"].value_counts().sort_index()
        total = len(batch_df)

        for size, count in size_counts.items():
            table.add_row(
                str(int(size)),
                str(count),
                f"{count / total * 100:.1f}%"
            )

        console.print(table)

        # Find incomplete batches
        if "embedding_timestamp" in batch_df.columns:
            # Group by timestamp and batch size to find batches
            batches = batch_df.groupby(["embedding_timestamp", "batch_size"]).size()

            incomplete = []
            for (timestamp, batch_size), count in batches.items():
                if count < batch_size:
                    incomplete.append({
                        "timestamp": timestamp,
                        "expected": int(batch_size),
                        "actual": int(count),
                        "utilization": count / batch_size * 100
                    })

            if incomplete:
                console.print(f"\n[yellow]Found {len(incomplete)} incomplete batches:[/yellow]")

                table = Table(title="Incomplete Batches", box=box.ROUNDED)
                table.add_column("Timestamp", style="cyan")
                table.add_column("Expected", style="green")
                table.add_column("Actual", style="yellow")
                table.add_column("Utilization", style="red")

                for batch in sorted(incomplete, key=lambda x: x["utilization"])[:10]:
                    table.add_row(
                        f"{batch['timestamp']:.2f}",
                        str(batch["expected"]),
                        str(batch["actual"]),
                        f"{batch['utilization']:.1f}%"
                    )

                console.print(table)
            else:
                console.print("\n[green]All batches are complete[/green]")

        # Average batch utilization
        avg_batch_size = batch_df["batch_size"].mean()
        console.print(f"\n[bold]Average batch size:[/bold] {avg_batch_size:.1f}")

        # Batch size by document
        if len(batch_df["document"].unique()) > 1:
            console.print("\n[bold]Batch size by document:[/bold]")
            for doc in batch_df["document"].unique():
                doc_avg = batch_df[batch_df["document"] == doc]["batch_size"].mean()
                console.print(f"  {doc}: {doc_avg:.1f}")


def main() -> int:
    """Main entry point for telemetry explorer."""
    parser = argparse.ArgumentParser(
        description="Interactive telemetry explorer for RagZoom benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore a single benchmark file
  python scripts/telemetry_explorer.py benchmark_results/metrics_200_tokens.json
  
  # Export data directly without interactive mode
  python scripts/telemetry_explorer.py benchmark_results/metrics_200_tokens.json --export nodes.csv
        """
    )

    parser.add_argument(
        "telemetry_file",
        type=Path,
        help="Path to telemetry JSON file"
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export all nodes to file (CSV or JSON based on extension) without interactive mode"
    )

    args = parser.parse_args()

    if not args.telemetry_file.exists():
        console.print(f"[red]Error: File not found: {args.telemetry_file}[/red]")
        return 1

    # Load telemetry data
    try:
        with open(args.telemetry_file) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        console.print(f"[red]Error: Invalid JSON: {e}[/red]")
        return 1

    if "telemetry" not in data:
        console.print("[red]Error: No telemetry data found in file[/red]")
        return 1

    # Create explorer
    explorer = TelemetryExplorer(data["telemetry"])

    # Handle direct export
    if args.export:
        if args.export.endswith(".csv"):
            explorer.nodes_df.to_csv(args.export, index=False)
            console.print(f"[green]Exported {len(explorer.nodes_df)} nodes to {args.export}[/green]")
        elif args.export.endswith(".json"):
            explorer.nodes_df.to_json(args.export, orient="records", indent=2)
            console.print(f"[green]Exported {len(explorer.nodes_df)} nodes to {args.export}[/green]")
        else:
            console.print("[red]Error: Export file must be .csv or .json[/red]")
            return 1
    else:
        # Run interactive session
        explorer.run_interactive_session()

    return 0


if __name__ == "__main__":
    sys.exit(main())
