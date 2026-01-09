#!/bin/bash
# Sync Claude Code transcript to RagZoom for historical context retrieval
#
# Input: JSON on stdin with session info including transcript_path
# Output: Silent on success
#
# Configuration:
#   data/.memory-env - contains MEMORY_ENV=<environment> (default: production)

set -euo pipefail

# Read JSON input from stdin
JSON=$(cat)

# Extract transcript path from JSON
TRANSCRIPT_PATH=$(echo "$JSON" | jq -r '.transcript_path // ""')

if [[ -n "$TRANSCRIPT_PATH" && -f "$TRANSCRIPT_PATH" ]]; then
    # Get repository root
    GIT_ROOT="$(git rev-parse --show-toplevel)"
    cd "$GIT_ROOT"

    # Read config from data/.memory-env
    if [[ -f "data/.memory-env" ]]; then
        source "data/.memory-env"
    fi
    MEMORY_ENV="${MEMORY_ENV:-production}"
    MEMORY_USER_ID="${MEMORY_USER_ID:-$USER}"

    # Discover gRPC address and export for CLI
    export RAGZOOM_SERVER_ADDRESS=$("$GIT_ROOT/scripts/get-grpc-address" "$MEMORY_ENV")
    export RAGZOOM_USER_ID="$MEMORY_USER_ID"

    # Run sync synchronously so errors are visible
    python -m ragzoom.cli sync-claude-code-transcript "$TRANSCRIPT_PATH"
fi
