#!/bin/bash
# Session start hook for Claude Code
#
# - Installs dependencies (gh CLI, Python packages) on first run in remote environments
# - Registers Claude Code session PID for MCP server lookup
#
# Input: JSON on stdin with session info including session_id
# Output: Silent on success

set -euo pipefail

# Read JSON input from stdin
JSON=$(cat)

# Get repository root
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Install dependencies only in remote (web) environments
if [[ "${CLAUDE_CODE_REMOTE:-}" == "true" ]]; then
    "$GIT_ROOT/scripts/install-dev-dependencies.sh"
fi

# Extract session ID from JSON
SESSION_ID=$(echo "$JSON" | jq -r '.session_id // ""')

if [[ -n "$SESSION_ID" ]]; then
    # Register the PID (PPID is Claude Code's PID)
    ragzoom-claude-code set-pid "$SESSION_ID" "$PPID"
fi
