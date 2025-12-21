#!/bin/bash
# Register Claude Code session PID for MCP server lookup
#
# Input: JSON on stdin with session info including session_id
# Output: Silent on success

set -euo pipefail

# Read JSON input from stdin
JSON=$(cat)

# Extract session ID from JSON
SESSION_ID=$(echo "$JSON" | jq -r '.session_id // ""')

if [[ -n "$SESSION_ID" ]]; then
    # Get repository root
    GIT_ROOT="$(git rev-parse --show-toplevel)"

    # Register the PID (PPID is Claude Code's PID)
    cd "$GIT_ROOT"
    python -m ragzoom.cli set-session-pid "$SESSION_ID" "$PPID"
fi

exit 0
