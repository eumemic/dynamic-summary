#!/bin/bash
# Quick test runner for development
# Usage: ./test_quick.sh [file_pattern]
# Examples:
#   ./test_quick.sh              # Run all tests
#   ./test_quick.sh splitter     # Run tests matching 'splitter'
#   ./test_quick.sh store        # Run tests matching 'store'

pattern=$1

if [ -z "$pattern" ]; then
    echo "Running all tests..."
    time pytest tests/ -v --tb=short
else
    echo "Running tests matching '$pattern'..."
    time pytest tests/ -k "$pattern" -v --tb=short
fi