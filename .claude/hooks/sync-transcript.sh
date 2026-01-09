#!/bin/bash
# Sync Claude Code transcript to RagZoom for historical context retrieval
#
# Input: JSON on stdin with session info including transcript_path
# Output: Silent on success
#
# Configuration:
#   RAGZOOM_ENV - which environment to sync to (default: production)
#   RAGZOOM_SERVER_ADDRESS - override address discovery (optional)

set -euo pipefail

# Read JSON input from stdin
JSON=$(cat)

# Extract transcript path from JSON
TRANSCRIPT_PATH=$(echo "$JSON" | jq -r '.transcript_path // ""')

if [[ -n "$TRANSCRIPT_PATH" && -f "$TRANSCRIPT_PATH" ]]; then
    # Get repository root
    GIT_ROOT="$(git rev-parse --show-toplevel)"
    cd "$GIT_ROOT"

    # Discover gRPC address if not already set
    if [[ -z "${RAGZOOM_SERVER_ADDRESS:-}" ]]; then
        RAGZOOM_ENV="${RAGZOOM_ENV:-production}"
        RAGZOOM_SERVER_ADDRESS=$("$GIT_ROOT/scripts/get-grpc-address" "$RAGZOOM_ENV")
        export RAGZOOM_SERVER_ADDRESS
    fi

    # Run sync synchronously so errors are visible
    python -m ragzoom.cli sync-claude-code-transcript "$TRANSCRIPT_PATH"
fi
