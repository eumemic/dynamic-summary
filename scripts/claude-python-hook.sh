#!/bin/bash
# Hook script for Claude to run Python quality checks on file changes
set -e  # Exit on any command failure

FILE_PATH="$1"

# Only process Python files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# Check if tools are available
if ! command -v black &> /dev/null; then
    echo "Warning: black not installed, skipping formatting"
else
    # Run black formatter (silent unless there's an error)
    black "$FILE_PATH" --quiet || echo "Black formatting failed for $FILE_PATH"
fi

if ! command -v ruff &> /dev/null; then
    echo "Warning: ruff not installed, skipping linting"
else
    # Run ruff linter with auto-fix (silent unless there's an error)
    ruff check "$FILE_PATH" --fix --quiet || echo "Ruff linting failed for $FILE_PATH"
fi

if ! command -v dmypy &> /dev/null; then
    echo "Warning: dmypy not installed, skipping type checking"
else
    # Run mypy type checker on the specific file
    # Check exit code instead of filtering output
    if ! dmypy run -- "$FILE_PATH" --ignore-missing-imports --no-error-summary --check-untyped-defs 2>&1; then
        # Type errors found - dmypy already printed them
        true  # Don't fail the hook, just report
    fi
fi