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
    cd "$GIT_ROOT"

    # Read RAGZOOM_SERVER_ADDRESS from .mcp.json if not already set
    if [[ -z "${RAGZOOM_SERVER_ADDRESS:-}" && -f ".mcp.json" ]]; then
        RAGZOOM_SERVER_ADDRESS=$(jq -r '.mcpServers["ragzoom-memory"].env.RAGZOOM_SERVER_ADDRESS // ""' .mcp.json)
        export RAGZOOM_SERVER_ADDRESS
    fi

    # Use separate state directory for remote server to avoid conflicts with local sync
    if [[ -n "${RAGZOOM_SERVER_ADDRESS:-}" && "$RAGZOOM_SERVER_ADDRESS" != "localhost"* ]]; then
        export RAGZOOM_STATE_DIR="data/transcript-state-remote"
    fi

    # Run sync synchronously so errors are visible
    python -m ragzoom.cli sync-claude-code-transcript "$TRANSCRIPT_PATH"
fi
