#!/bin/bash
# Sync Claude Code transcript to RagZoom for historical context retrieval
#
# Input: JSON on stdin with session info including transcript_path
# Output: Silent on success

set -euo pipefail

# Read JSON input from stdin
JSON=$(cat)

# Extract transcript path from JSON
TRANSCRIPT_PATH=$(echo "$JSON" | jq -r '.transcript_path // ""')

if [[ -n "$TRANSCRIPT_PATH" && -f "$TRANSCRIPT_PATH" ]]; then
    # Get repository root
    GIT_ROOT="$(git rev-parse --show-toplevel)"

    # Run sync synchronously so errors are visible
    cd "$GIT_ROOT"
    python -m ragzoom.cli sync-claude-code-transcript "$TRANSCRIPT_PATH"
fi
