#!/bin/bash
# Quick test runner for development
# Usage: ./test_quick.sh [file_pattern]
# Examples:
#   ./test_quick.sh              # Run all tests
#   ./test_quick.sh splitter     # Run tests matching 'splitter'
#   ./test_quick.sh store        # Run tests matching 'store'

pattern=$1

if [ -z "$pattern" ]; then
    echo "Running all tests (excluding benchmarks)..."
    time pytest tests/ -v --tb=short -m "not benchmark"
else
    echo "Running tests matching '$pattern' (excluding benchmarks)..."
    time pytest tests/ -k "$pattern" -v --tb=short -m "not benchmark"
fi