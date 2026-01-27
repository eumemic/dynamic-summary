#!/bin/bash
# Sync Claude Code transcript to RagZoom for historical context retrieval
#
# Input: JSON on stdin with session info including transcript_path
# Output: Silent on success

set -euo pipefail

# Skip hooks when sync is disabled (e.g., RAGZOOM_DISABLE_SYNC=1 ralph build)
if [[ "${RAGZOOM_DISABLE_SYNC:-}" == "1" ]]; then
    cat > /dev/null  # consume stdin
    exit 0
fi

# Read JSON input from stdin
JSON=$(cat)

# Extract transcript path from JSON
TRANSCRIPT_PATH=$(echo "$JSON" | jq -r '.transcript_path // ""')

if [[ -n "$TRANSCRIPT_PATH" && -f "$TRANSCRIPT_PATH" ]]; then
    # Get repository root
    GIT_ROOT="$(git rev-parse --show-toplevel)"

    # Run sync synchronously so errors are visible
    cd "$GIT_ROOT"
    ragzoom-claude-code sync "$TRANSCRIPT_PATH"
fi
