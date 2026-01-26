#!/bin/bash
# Claude PostToolUse hook for running Python quality checks
# This hook is called after Write, Edit, or MultiEdit operations
#
# Input: JSON on stdin with tool information
# Output: Silent on success, errors to stderr with exit code 2

set -euo pipefail

# Skip hooks when sync is disabled (e.g., RAGZOOM_DISABLE_SYNC=1 ralph build)
if [[ "${RAGZOOM_DISABLE_SYNC:-}" == "1" ]]; then
    cat > /dev/null  # consume stdin
    exit 0
fi

# Start timing
START_TIME=$(date +%s%N)

# Read JSON input from stdin
JSON=$(cat)

# Extract file path from JSON (handle both file_path and filePath)
FILE_PATH=$(echo "$JSON" | jq -r '.tool_input.file_path // .tool_input.filePath // ""')

# Only process Python files
if [[ "$FILE_PATH" == *.py ]]; then
    # Get repository root (works in main repo and worktrees)
    GIT_ROOT="$(git rev-parse --show-toplevel)"
    
    # Run checks, skipping tests, jscpd, and bandit for speed
    # Use --fail-fast to stop at first error (saves tokens)
    # Ignore F401 (unused imports) to prevent removal between agent edits
    # Pass the specific file that was edited
    "$GIT_ROOT/scripts/run-checks.sh" --skip tests,jscpd,bandit --fail-fast --ignore-lint-rules F401 "$FILE_PATH"
    RESULT=$?
    
    # Calculate elapsed time in milliseconds
    END_TIME=$(date +%s%N)
    ELAPSED=$(( ($END_TIME - $START_TIME) / 1000000 ))
    
    # Warn if checks took too long
    if [ $ELAPSED -gt 1000 ]; then
        echo "⚠️  Python checks took ${ELAPSED}ms (>1s threshold)" >&2
    fi
    
    # Exit with the result from checks
    exit $RESULT
else
    # Not a Python file, exit successfully
    exit 0
fi