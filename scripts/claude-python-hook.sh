#!/bin/bash
# Hook script for Claude to run Python quality checks on file changes

FILE_PATH="$1"

# Only process Python files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# Run black formatter (silent unless there's an error)
black "$FILE_PATH" --quiet

# Run ruff linter with auto-fix (silent unless there's an error)
ruff check "$FILE_PATH" --fix --quiet

# Run mypy type checker (filter out success messages)
dmypy run -- ragzoom --ignore-missing-imports --no-error-summary --check-untyped-defs 2>&1 | grep -v "Success: no issues found" || true