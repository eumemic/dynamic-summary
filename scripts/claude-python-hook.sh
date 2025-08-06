#!/bin/bash
# Hook script for Claude to run Python quality checks on file changes

FILE_PATH="$1"

# Only process Python files
if [[ "$FILE_PATH" != *.py ]]; then
    exit 0
fi

# Source common Python check functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/python-checks-common.sh"

# Run Python checks in parallel
# Pass single file, no modified files check needed, and exit_on_failure=false for non-blocking
run_python_checks_parallel "$FILE_PATH" "" "false"

# Always exit successfully to avoid blocking Claude operations
exit 0