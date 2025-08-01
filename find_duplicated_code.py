#!/usr/bin/env python3
"""Find code duplications using jscpd and filter true positives with Claude."""

import argparse
import asyncio
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


async def analyze_duplicate_batch(duplicates: List[Dict], batch_id: int, threshold: float = 3.5) -> List[Tuple[Dict, bool, str]]:
    """Analyze a batch of duplicates using Claude Sonnet."""
    results = []
    
    # Create prompt for this batch
    prompt = f"""Rate each code duplication on a scale of 1-5:
1 = Definitely not duplication (false positive)
2 = Probably not duplication  
3 = Borderline/uncertain
4 = Probably duplication
5 = Definitely duplication (true positive)

For each duplicate, respond with ONLY:
<number>. <rating> - <one-line reason>

Duplicates to analyze:
"""
    
    for i, dup in enumerate(duplicates, 1):
        file1 = dup['firstFile']['name']
        file2 = dup['secondFile']['name']
        lines = dup['lines']
        fragment = dup['fragment'][:500]  # Limit fragment size
        
        prompt += f"""
{i}. {file1} (lines {dup['firstFile']['start']}-{dup['firstFile']['end']}) <-> {file2} (lines {dup['secondFile']['start']}-{dup['secondFile']['end']})
Lines: {lines}
Fragment:
{fragment}
{"..." if len(dup['fragment']) > 500 else ""}
---
"""
    
    # Run claude with Sonnet model
    cmd = ['claude', '--model', 'sonnet', '-p', prompt]
    
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            print(f"Batch {batch_id} failed: {stderr.decode()}", file=sys.stderr)
            for dup in duplicates:
                results.append((dup, False, "Claude analysis failed"))
        else:
            # Parse Claude's response
            response = stdout.decode()
            lines = response.strip().split('\n')
            
            # Extract ratings
            for i, dup in enumerate(duplicates, 1):
                found = False
                for line in lines:
                    if line.strip().startswith(f"{i}."):
                        # Try to parse rating
                        import re
                        # Handle <rating> tags if present
                        clean_line = line.strip()
                        if '<rating>' in clean_line and '</rating>' in clean_line:
                            # Extract content between tags
                            start = clean_line.find('<rating>') + 8
                            end = clean_line.find('</rating>')
                            clean_line = f"{i}. {clean_line[start:end].strip()}"
                        
                        match = re.match(rf'^{i}\.\s*(\d+)\s*[-–]\s*(.+)$', clean_line)
                        if match:
                            rating = int(match.group(1))
                            reason = f"{rating} - {match.group(2)}"
                            is_positive = rating >= threshold
                            results.append((dup, is_positive, reason))
                            found = True
                            break
                
                if not found:
                    results.append((dup, False, "Could not parse rating"))
                    
    except Exception as e:
        print(f"Batch {batch_id} error: {e}", file=sys.stderr)
        for dup in duplicates:
            results.append((dup, False, f"Error: {str(e)}"))
    
    return results


async def analyze_duplicates_parallel(duplicates: List[Dict], n_parallel: int, threshold: float = 3.5) -> List[Dict]:
    """Analyze duplicates using N parallel Claude instances."""
    if not duplicates:
        return []
    
    # Split duplicates into n_parallel batches as evenly as possible
    batches = []
    base_size = len(duplicates) // n_parallel
    remainder = len(duplicates) % n_parallel
    
    start = 0
    for i in range(n_parallel):
        # First 'remainder' batches get base_size + 1 items
        batch_size = base_size + (1 if i < remainder else 0)
        if batch_size > 0:
            batch = duplicates[start:start + batch_size]
            batches.append(batch)
            start += batch_size
    
    print(f"Analyzing {len(duplicates)} potential duplicates with {len(batches)} parallel Claude instances...", file=sys.stderr)
    
    # Run all batches in parallel
    tasks = [analyze_duplicate_batch(batch, i+1, threshold) for i, batch in enumerate(batches)]
    batch_results = await asyncio.gather(*tasks)
    
    # Collect only true positives
    true_positives = []
    for batch_result in batch_results:
        for dup, is_true_positive, reason in batch_result:
            if is_true_positive:
                dup['claude_reason'] = reason
                true_positives.append(dup)
    
    return true_positives


def run_jscpd(paths: List[str], min_lines: int, min_tokens: int) -> Dict:
    """Run jscpd and return the JSON report."""
    cmd = [
        'npx', 'jscpd',
        *paths,
        '--min-lines', str(min_lines),
        '--min-tokens', str(min_tokens),
        '--reporters', 'json',
        '--silent',
        '--threshold', '100'  # Allow any amount of duplication
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print(f"jscpd failed: {e.stderr.decode()}", file=sys.stderr)
        sys.exit(1)
    
    # jscpd writes to jscpd-report.json by default
    try:
        with open('jscpd-report.json', 'r') as f:
            data = json.load(f)
        return data
    finally:
        Path('jscpd-report.json').unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description='Find code duplications and filter true positives with Claude'
    )
    parser.add_argument('paths', nargs='+', help='Paths to analyze')
    parser.add_argument('--min-lines', type=int, default=12,
                        help='Minimum lines for duplication (default: 12)')
    parser.add_argument('--min-tokens', type=int, default=16,
                        help='Minimum tokens for duplication (default: 16)')
    parser.add_argument('--num-triagers', type=int, default=4,
                        help='Number of parallel Claude instances for triaging (default: 4)')
    parser.add_argument('--output', choices=['json', 'text'], default='text',
                        help='Output format (default: text)')
    parser.add_argument('--show-code', action='store_true',
                        help='Show code snippets for each duplication')
    parser.add_argument('--threshold', type=float, default=3.5,
                        help='Rating threshold for true positives (1-5, default: 3.5)')
    
    args = parser.parse_args()
    
    # Run jscpd
    report = run_jscpd(args.paths, args.min_lines, args.min_tokens)
    
    duplicates = report.get('duplicates', [])
    if not duplicates:
        print("No duplications found!", file=sys.stderr)
        return
    
    # Analyze with Claude
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    true_positives = loop.run_until_complete(
        analyze_duplicates_parallel(duplicates, args.num_triagers, args.threshold)
    )
    
    print(f"\nFound {len(true_positives)} true positive duplications, {len(duplicates) - len(true_positives)} false positives filtered out", file=sys.stderr)
    
    # Output results
    if args.output == 'json':
        json.dump({
            'duplicates': true_positives,
            'summary': {
                'total_analyzed': len(duplicates),
                'true_positives': len(true_positives),
                'false_positives': len(duplicates) - len(true_positives)
            }
        }, sys.stdout, indent=2)
    else:
        # Text output
        if true_positives:
            for i, dup in enumerate(true_positives, 1):
                file1 = dup['firstFile']['name']
                file2 = dup['secondFile']['name']
                lines1 = f"{dup['firstFile']['start']}-{dup['firstFile']['end']}"
                lines2 = f"{dup['secondFile']['start']}-{dup['secondFile']['end']}"
                
                print(f"\n{i}. {file1}:{lines1} <-> {file2}:{lines2}")
                # Extract just the reason without the classification prefix
                reason = dup.get('claude_reason', 'No reason provided')
                if ': ' in reason:
                    reason = reason.split(': ', 1)[1]
                print(f"   {reason}")
                
                if args.show_code:
                    print("\n```")
                    # Show the full fragment, not truncated
                    print(dup['fragment'].strip())
                    print("```")
        else:
            print("\nNo true positive duplications found.")


if __name__ == '__main__':
    main()