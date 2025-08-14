#!/usr/bin/env python3
"""Extract problematic summarization cases from database."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ragzoom.config import IndexConfig

@dataclass
class ProblemCase:
    """A problematic summarization case."""
    node_id: str
    height: int
    left_text: str
    right_text: str
    summary: str
    actual_tokens: int
    target_tokens: int
    divergence: int
    divergence_pct: float
    document_id: Optional[str] = None
    preceding_context: Optional[str] = None

def extract_problem_cases(
    telemetry_path: Path,
    db_path: Path,
    max_cases: int = 50
):
    """Extract problematic cases from database using telemetry config.
    
    Args:
        telemetry_path: Path to telemetry JSON (for config only)
        db_path: Path to ragzoom.db
        max_cases: Maximum cases to include
    """
    
    # Step 1: Get config and nodes from telemetry
    with open(telemetry_path) as f:
        telemetry = json.load(f)
    
    config_dict = telemetry.get("config", {})
    config = IndexConfig.from_dict(config_dict)
    target_tokens = config.target_chunk_tokens
    
    # Get document ID from telemetry
    document_id = telemetry.get("document_id")
    if not document_id:
        raise ValueError(f"No document_id found in telemetry file {telemetry_path}")
    
    # Extract all node IDs from telemetry for validation
    # Even with same document, database could have nodes from a different indexing run
    telemetry_nodes = telemetry.get("nodes", [])
    telemetry_node_ids = {node.get("node_id") for node in telemetry_nodes if node.get("node_id")}
    
    # Step 2: Query database for worst divergences
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    
    query = """
    SELECT 
        n.*,
        (n.token_count - ?) as divergence,
        (n.token_count - ?) * 100.0 / ? as divergence_pct
    FROM tree_nodes n
    WHERE n.left_child_id IS NOT NULL  -- Non-leaf nodes only
      AND n.token_count IS NOT NULL
      AND n.document_id = ?  -- Only nodes from this document
    ORDER BY ABS(n.token_count - ?) DESC  -- Order by absolute divergence
    LIMIT 500  -- Get more than we need for sampling
    """
    # Note: The * selector will automatically include preceding_neighbor_id if it exists
    
    cursor = conn.execute(query, (target_tokens, target_tokens, target_tokens, document_id, target_tokens))
    problem_nodes = cursor.fetchall()
    
    # Build cases with full context
    cases = []
    skipped_not_in_telemetry = 0
    
    for node in problem_nodes:
        # Skip nodes not in telemetry (from different indexing run)
        if node["id"] not in telemetry_node_ids:
            skipped_not_in_telemetry += 1
            continue
            
        # Get children for input texts
        left_child = get_node(conn, node["left_child_id"])
        right_child = get_node(conn, node["right_child_id"])
        
        if not left_child or not right_child:
            continue
        
        # Calculate height
        height = calculate_height(conn, node["id"])
        
        # Get preceding context from the preceding neighbor node
        preceding_context = None
        try:
            preceding_neighbor_id = node["preceding_neighbor_id"]
            if preceding_neighbor_id:
                preceding_node = get_node(conn, preceding_neighbor_id)
                if preceding_node:
                    preceding_context = preceding_node["text"]
        except (KeyError, IndexError) as e:
            # Column doesn't exist in this database yet
            raise ValueError(
                "Database does not have preceding_neighbor_id column. "
                "Please re-index the document to populate this field."
            ) from e
        
        case = ProblemCase(
            node_id=node["id"],
            height=height,
            left_text=left_child["text"],
            right_text=right_child["text"],
            summary=node["text"],
            actual_tokens=node["token_count"],
            target_tokens=target_tokens,
            divergence=node["divergence"],
            divergence_pct=node["divergence_pct"],
            document_id=node["document_id"],
            preceding_context=preceding_context
        )
        cases.append(case)
    
    if skipped_not_in_telemetry > 0:
        print(f"Warning: Skipped {skipped_not_in_telemetry} nodes not found in telemetry (likely from different run)")
    
    # Just take the top N most divergent cases
    # They're already sorted by divergence from the SQL query
    selected_cases = cases[:max_cases]
    
    # Prepare output filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("tests/summarization_problem_cases")
    output_path = output_dir / f"problem_cases_{timestamp}.json"
    
    # Save output
    output = {
        "metadata": {
            "extraction_date": datetime.now().isoformat(),
            "database_path": str(db_path),
            "telemetry_path": str(telemetry_path),
            "config": config_dict,  # Save as dict for JSON serialization
            "total_problem_nodes": len(cases),
            "cases_included": len(selected_cases)
        },
        "cases": [asdict(c) for c in selected_cases]
    }
    
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Print summary
    print(f"Extracted {len(selected_cases)} problem cases (from {len(cases)} total):")
    print(f"  Verbatim (>150%): {sum(1 for c in selected_cases if c.divergence_pct > 150)}")
    print(f"  Severe overshoot (50-150%): {sum(1 for c in selected_cases if 50 < c.divergence_pct <= 150)}")
    print(f"  Severe undershoot (<-30%): {sum(1 for c in selected_cases if c.divergence_pct < -30)}")
    print(f"  Moderate (other): {sum(1 for c in selected_cases if -30 <= c.divergence_pct <= 50)}")
    print(f"  Output: {output_path}")
    
    return selected_cases

def get_node(conn, node_id):
    """Get node from database."""
    if not node_id:
        return None
    cursor = conn.execute("SELECT * FROM tree_nodes WHERE id = ?", (node_id,))
    return cursor.fetchone()

def calculate_height(conn, node_id):
    """Calculate height recursively."""
    node = get_node(conn, node_id)
    if not node or (not node["left_child_id"] and not node["right_child_id"]):
        return 0
    
    left_height = calculate_height(conn, node["left_child_id"]) if node["left_child_id"] else 0
    right_height = calculate_height(conn, node["right_child_id"]) if node["right_child_id"] else 0
    
    return 1 + max(left_height, right_height)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("telemetry", type=Path, help="Telemetry JSON file")
    parser.add_argument("--db", type=Path, default=Path("ragzoom.db"))
    parser.add_argument("--max-cases", type=int, default=50)
    
    args = parser.parse_args()
    
    extract_problem_cases(
        telemetry_path=args.telemetry,
        db_path=args.db,
        max_cases=args.max_cases
    )

if __name__ == "__main__":
    main()