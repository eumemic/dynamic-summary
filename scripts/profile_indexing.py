#!/usr/bin/env python3
"""Profile indexing performance to identify bottlenecks."""

import cProfile
import pstats
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ragzoom.cli import index
import click

def profile_indexing():
    """Profile the indexing command with The Hobbit at 50 tokens."""
    
    # Import the CLI properly
    from click.testing import CliRunner
    from ragzoom.cli import cli
    
    runner = CliRunner()
    
    # Run the index command with the same parameters
    result = runner.invoke(cli, [
        'index', 'test_data/the_hobbit.txt',
        '--target-chunk-tokens', '50',
        '--validate',
        '--debug',
        '--no-progress'
    ])
    
    if result.exit_code != 0:
        print(f"Command failed with exit code {result.exit_code}")
        print("Output:", result.output)
        if result.exception:
            raise result.exception

if __name__ == "__main__":
    profiler = cProfile.Profile()
    
    print("Starting profiling of indexing process...")
    print("This will index The Hobbit with 50-token chunks...")
    print("-" * 60)
    
    profiler.enable()
    try:
        profile_indexing()
    finally:
        profiler.disable()
    
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)
    
    # Print stats sorted by cumulative time
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\nTop 30 functions by cumulative time:")
    stats.print_stats(30)
    
    # Also show by total time
    print("\n" + "=" * 60)
    print("Top 20 functions by total time spent in function:")
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    # Save detailed profile for analysis
    stats.dump_stats('indexing_profile.stats')
    print(f"\nDetailed profile saved to indexing_profile.stats")
    print("View with: python -m pstats indexing_profile.stats")