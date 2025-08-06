#!/bin/bash
# Hook script for Claude to run Python quality checks on file changes
# Note: We don't use set -e because we want to run all tools even if one fails

FILE_PATH="$1"

# Only process Python files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# Check if tools are available and run them
if ! command -v black &> /dev/null; then
    echo "Warning: black not installed, skipping formatting"
else
    # Run black formatter (silent unless there's an error)
    if ! black "$FILE_PATH" --quiet; then
        echo "Black: formatting issues in $FILE_PATH"
    fi
fi

if ! command -v ruff &> /dev/null; then
    echo "Warning: ruff not installed, skipping linting"
else
    # Run ruff linter with auto-fix (silent unless there's an error)
    if ! ruff check "$FILE_PATH" --fix --quiet; then
        echo "Ruff: linting issues in $FILE_PATH"
    fi
fi

if ! command -v dmypy &> /dev/null; then
    echo "Warning: dmypy not installed, skipping type checking"
else
    # Run mypy type checker on the specific file
    if ! dmypy run -- "$FILE_PATH" --ignore-missing-imports --no-error-summary --check-untyped-defs 2>&1; then
        # Type errors found - dmypy already printed them
        echo "Mypy: type errors in $FILE_PATH"
    fi
fi

# Always exit successfully to avoid blocking Claude operations
exit 0