#!/usr/bin/env python3
"""Validate telemetry data format and integrity.

This script checks telemetry data for format compliance, data integrity,
and common issues that might affect analysis accuracy.

Usage:
    python scripts/validate_telemetry.py benchmark_results/metrics_200_tokens.json
    python scripts/validate_telemetry.py benchmark_results/ --fix
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Add parent directory to path for importing ragzoom
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragzoom.telemetry import SUPPORTED_TELEMETRY_VERSIONS


class TelemetryValidator:
    """Validates telemetry data format and integrity."""

    def __init__(self, fix_issues: bool = False) -> None:
        """Initialize validator.
        
        Args:
            fix_issues: If True, attempt to fix minor issues
        """
        self.fix_issues = fix_issues
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.fixed: List[str] = []

    def validate_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Validate a single telemetry file.
        
        Args:
            file_path: Path to JSON file containing telemetry
            
        Returns:
            Tuple of (is_valid, data) where data may be fixed version
        """
        self.errors = []
        self.warnings = []
        self.fixed = []

        print(f"\nValidating {file_path.name}...")

        try:
            with open(file_path) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.errors.append(f"Invalid JSON: {e}")
            return False, {}
        except Exception as e:
            self.errors.append(f"Cannot read file: {e}")
            return False, {}

        # Check if file contains telemetry
        if "telemetry" not in data:
            self.warnings.append("No telemetry data found in file")
            return True, data

        telemetry = data["telemetry"]

        # Validate telemetry structure
        self._validate_format_version(telemetry)
        self._validate_documents(telemetry)
        self._validate_nodes(telemetry)
        self._validate_consistency(telemetry)

        # Report results
        is_valid = len(self.errors) == 0

        if self.errors:
            print(f"❌ Found {len(self.errors)} errors:")
            for error in self.errors:
                print(f"   - {error}")

        if self.warnings:
            print(f"⚠️  Found {len(self.warnings)} warnings:")
            for warning in self.warnings:
                print(f"   - {warning}")

        if self.fixed:
            print(f"✅ Fixed {len(self.fixed)} issues:")
            for fix in self.fixed:
                print(f"   - {fix}")

        if is_valid and not self.warnings and not self.fixed:
            print("✅ Telemetry data is valid")

        return is_valid, data

    def _validate_format_version(self, telemetry: Dict[str, Any]) -> None:
        """Validate format version."""
        if "format_version" not in telemetry:
            if self.fix_issues:
                telemetry["format_version"] = "1.0"
                self.fixed.append("Added missing format_version (1.0)")
            else:
                self.errors.append("Missing format_version")
        else:
            version = telemetry["format_version"]
            if version not in SUPPORTED_TELEMETRY_VERSIONS:
                self.errors.append(
                    f"Unsupported format version: {version}. "
                    f"Supported versions: {SUPPORTED_TELEMETRY_VERSIONS}"
                )

    def _validate_documents(self, telemetry: Dict[str, Any]) -> None:
        """Validate documents structure."""
        if "documents" not in telemetry:
            self.errors.append("Missing documents section")
            return

        documents = telemetry["documents"]
        if not isinstance(documents, dict):
            self.errors.append("Documents must be a dictionary")
            return

        if not documents:
            self.warnings.append("No documents in telemetry")

    def _validate_nodes(self, telemetry: Dict[str, Any]) -> None:
        """Validate individual nodes."""
        documents = telemetry.get("documents", {})

        for doc_type, doc_data in documents.items():
            if not isinstance(doc_data, dict):
                self.errors.append(f"Document '{doc_type}' must be a dictionary")
                continue

            nodes = doc_data.get("nodes", [])
            if not isinstance(nodes, list):
                self.errors.append(f"Nodes in '{doc_type}' must be a list")
                continue

            # Validate each node
            for i, node in enumerate(nodes):
                self._validate_node(node, f"{doc_type}[{i}]")

            # Validate metadata if present
            metadata = doc_data.get("metadata", {})
            if metadata:
                self._validate_metadata(metadata, nodes, doc_type)

    def _validate_node(self, node: Dict[str, Any], path: str) -> None:
        """Validate a single node."""
        # Required fields
        required_fields = ["node_id", "node_type", "level", "span", "created_at"]
        for field in required_fields:
            if field not in node:
                self.errors.append(f"Node {path} missing required field: {field}")

        # Validate node_type
        if "node_type" in node:
            if node["node_type"] not in ["leaf", "summary"]:
                self.errors.append(
                    f"Node {path} has invalid node_type: {node['node_type']}"
                )

        # Validate level
        if "level" in node:
            if not isinstance(node["level"], int) or node["level"] < 0:
                self.errors.append(f"Node {path} has invalid level: {node['level']}")

        # Validate span
        if "span" in node:
            span = node["span"]
            if not isinstance(span, list) or len(span) != 2:
                self.errors.append(f"Node {path} has invalid span format")
            elif not all(isinstance(x, int) for x in span):
                self.errors.append(f"Node {path} span must contain integers")
            elif span[0] > span[1]:
                self.errors.append(f"Node {path} span start > end")

        # Validate embedding if present
        if "embedding" in node and node["embedding"]:
            self._validate_embedding(node["embedding"], f"{path}.embedding")

        # Validate summary attempts if present
        if "summary_attempts" in node and node["summary_attempts"]:
            for j, attempt in enumerate(node["summary_attempts"]):
                self._validate_summary_attempt(attempt, f"{path}.summary_attempts[{j}]")

    def _validate_embedding(self, embedding: Dict[str, Any], path: str) -> None:
        """Validate embedding telemetry."""
        required = ["text_tokens", "batch_size", "batch_position", "model", "timestamp"]
        for field in required:
            if field not in embedding:
                self.errors.append(f"{path} missing required field: {field}")

        # Validate numeric fields
        if "text_tokens" in embedding and embedding["text_tokens"] <= 0:
            self.warnings.append(f"{path} has zero or negative text_tokens")

        if "batch_size" in embedding and embedding["batch_size"] <= 0:
            self.errors.append(f"{path} has invalid batch_size")

        if "batch_position" in embedding and "batch_size" in embedding:
            if embedding["batch_position"] >= embedding["batch_size"]:
                self.errors.append(f"{path} batch_position >= batch_size")

    def _validate_summary_attempt(self, attempt: Dict[str, Any], path: str) -> None:
        """Validate summary attempt telemetry."""
        required = [
            "is_retry", "target_tokens", "input_text_tokens",
            "prompt_tokens", "completion_tokens", "actual_tokens",
            "status", "model", "timestamp"
        ]

        for field in required:
            if field not in attempt:
                self.errors.append(f"{path} missing required field: {field}")

        # Validate status
        if "status" in attempt:
            valid_statuses = ["accepted", "rejected_over", "rejected_under", "error"]
            if attempt["status"] not in valid_statuses:
                self.errors.append(
                    f"{path} has invalid status: {attempt['status']}"
                )

        # Validate token counts
        token_fields = [
            "target_tokens", "input_text_tokens",
            "prompt_tokens", "completion_tokens", "actual_tokens"
        ]
        for field in token_fields:
            if field in attempt and attempt[field] < 0:
                self.errors.append(f"{path} has negative {field}")

        # Logical validations
        if "prompt_tokens" in attempt and "input_text_tokens" in attempt:
            if attempt["prompt_tokens"] < attempt["input_text_tokens"]:
                self.warnings.append(
                    f"{path} prompt_tokens < input_text_tokens "
                    "(might indicate missing system prompt)"
                )

        # Check rejection reasons
        if "status" in attempt and attempt["status"].startswith("rejected"):
            if not attempt.get("rejection_reason"):
                self.warnings.append(
                    f"{path} is rejected but has no rejection_reason"
                )

    def _validate_metadata(self, metadata: Dict[str, Any], nodes: List[Dict[str, Any]], doc_type: str) -> None:
        """Validate document metadata against actual nodes."""
        # Count actual nodes
        actual_total = len(nodes)
        actual_leaf = sum(1 for n in nodes if n.get("node_type") == "leaf")
        actual_summary = sum(1 for n in nodes if n.get("node_type") == "summary")

        # Check totals
        if "total_nodes" in metadata:
            if metadata["total_nodes"] != actual_total:
                if self.fix_issues:
                    metadata["total_nodes"] = actual_total
                    self.fixed.append(f"Fixed total_nodes in {doc_type}")
                else:
                    self.errors.append(
                        f"Metadata mismatch in {doc_type}: "
                        f"total_nodes={metadata['total_nodes']} but found {actual_total}"
                    )

        if "leaf_nodes" in metadata:
            if metadata["leaf_nodes"] != actual_leaf:
                if self.fix_issues:
                    metadata["leaf_nodes"] = actual_leaf
                    self.fixed.append(f"Fixed leaf_nodes in {doc_type}")
                else:
                    self.errors.append(
                        f"Metadata mismatch in {doc_type}: "
                        f"leaf_nodes={metadata['leaf_nodes']} but found {actual_leaf}"
                    )

        if "summary_nodes" in metadata:
            if metadata["summary_nodes"] != actual_summary:
                if self.fix_issues:
                    metadata["summary_nodes"] = actual_summary
                    self.fixed.append(f"Fixed summary_nodes in {doc_type}")
                else:
                    self.errors.append(
                        f"Metadata mismatch in {doc_type}: "
                        f"summary_nodes={metadata['summary_nodes']} but found {actual_summary}"
                    )

    def _validate_consistency(self, telemetry: Dict[str, Any]) -> None:
        """Validate internal consistency of telemetry data."""
        documents = telemetry.get("documents", {})

        for doc_type, doc_data in documents.items():
            nodes = doc_data.get("nodes", [])

            # Check node IDs are unique
            node_ids: Set[str] = set()
            for node in nodes:
                node_id = node.get("node_id")
                if node_id:
                    if node_id in node_ids:
                        self.errors.append(f"Duplicate node_id in {doc_type}: {node_id}")
                    node_ids.add(node_id)

            # Check level consistency
            leaf_level = None
            for node in nodes:
                if node.get("node_type") == "leaf":
                    level = node.get("level", -1)
                    if leaf_level is None:
                        leaf_level = level
                    elif level != leaf_level:
                        self.warnings.append(
                            f"Inconsistent leaf levels in {doc_type}: {level} vs {leaf_level}"
                        )

            # Check embedding batch consistency
            batch_map: Dict[Tuple[float, int], Set[int]] = {}
            for node in nodes:
                if "embedding" in node and node["embedding"]:
                    emb = node["embedding"]
                    timestamp = emb.get("timestamp")
                    batch_size = emb.get("batch_size")
                    batch_pos = emb.get("batch_position", -1)

                    if timestamp and batch_size:
                        key = (timestamp, batch_size)
                        if key not in batch_map:
                            batch_map[key] = set()

                        if batch_pos in batch_map[key]:
                            self.errors.append(
                                f"Duplicate batch position in {doc_type}: "
                                f"position {batch_pos} at timestamp {timestamp}"
                            )
                        batch_map[key].add(batch_pos)

            # Verify batch completeness
            for (timestamp, batch_size), positions in batch_map.items():
                expected = set(range(batch_size))
                if positions != expected:
                    missing = expected - positions
                    if missing:
                        self.warnings.append(
                            f"Incomplete batch in {doc_type} at {timestamp}: "
                            f"missing positions {sorted(missing)}"
                        )


def validate_directory(dir_path: Path, validator: TelemetryValidator) -> None:
    """Validate all telemetry files in a directory."""
    json_files = list(dir_path.glob("metrics_*_tokens.json"))

    if not json_files:
        print(f"No benchmark files found in {dir_path}")
        return

    print(f"Found {len(json_files)} benchmark files")

    all_valid = True
    fixed_files = []
    validation_results = {}  # Store results to avoid re-validation

    for file in sorted(json_files):
        is_valid, data = validator.validate_file(file)
        validation_results[file] = is_valid

        if not is_valid:
            all_valid = False

        # Save fixed file if requested
        if validator.fix_issues and validator.fixed:
            fixed_files.append(file)
            if is_valid:  # Only save if all errors were fixed
                with open(file, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"   Saved fixed version to {file}")

    # Summary
    print(f"\n{'='*60}")
    print("Validation Summary:")
    print(f"  Total files: {len(json_files)}")
    # Count valid files from our stored results
    valid_count = sum(1 for is_valid in validation_results.values() if is_valid)
    print(f"  Valid files: {valid_count}")
    if validator.fix_issues and fixed_files:
        print(f"  Fixed files: {len(fixed_files)}")
    print(f"  Overall: {'✅ PASS' if all_valid else '❌ FAIL'}")


def main() -> int:
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description="Validate telemetry data format and integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a single file
  python scripts/validate_telemetry.py benchmark_results/metrics_200_tokens.json
  
  # Validate all files in directory
  python scripts/validate_telemetry.py benchmark_results/
  
  # Fix minor issues automatically
  python scripts/validate_telemetry.py benchmark_results/ --fix
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Input file or directory containing telemetry JSON files"
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix minor issues (updates files in place)"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as errors"
    )

    args = parser.parse_args()

    # Create validator
    validator = TelemetryValidator(fix_issues=args.fix)

    # Process input
    if args.input.is_file():
        is_valid, _ = validator.validate_file(args.input)

        if args.strict and validator.warnings:
            is_valid = False

        return 0 if is_valid else 1

    elif args.input.is_dir():
        validate_directory(args.input, validator)
        return 0

    else:
        print(f"Error: {args.input} not found")
        return 1


if __name__ == "__main__":
    sys.exit(main())

