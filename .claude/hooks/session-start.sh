#!/bin/bash
# Session start hook for Claude Code
#
# - Installs dependencies (gh CLI, Python packages) on first run in remote environments
# - Registers Claude Code session PID for MCP server lookup
# - Injects session ID and transcript path into context
#
# Input: JSON on stdin with session info including session_id, cwd, trigger
# Output: JSON with systemMessage containing session info

set -euo pipefail

# Skip hooks when sync is disabled (e.g., RAGZOOM_DISABLE_SYNC=1 ralph build)
if [[ "${RAGZOOM_DISABLE_SYNC:-}" == "1" ]]; then
    cat > /dev/null  # consume stdin
    exit 0
fi

# Read JSON input from stdin
JSON=$(cat)

# Get repository root
GIT_ROOT="$(git rev-parse --show-toplevel)"

# Install dependencies only in remote (web) environments
if [[ "${CLAUDE_CODE_REMOTE:-}" == "true" ]]; then
    "$GIT_ROOT/scripts/install-dev-dependencies.sh"
fi

# Extract fields from JSON
SESSION_ID=$(echo "$JSON" | jq -r '.session_id // ""')
CWD=$(echo "$JSON" | jq -r '.cwd // ""')
TRIGGER=$(echo "$JSON" | jq -r '.trigger // ""')

if [[ -n "$SESSION_ID" ]]; then
    # Register the PID (PPID is Claude Code's PID)
    ragzoom-claude-code set-pid "$SESSION_ID" "$PPID" >/dev/null 2>&1 || true

    # Derive transcript path: ~/.claude/projects/<encoded-cwd>/<session-id>.jsonl
    # CWD like /Users/tom/code/foo -> -Users-tom-code-foo
    ENCODED_CWD=$(echo "$CWD" | sed 's|/|-|g')
    TRANSCRIPT_PATH="$HOME/.claude/projects/$ENCODED_CWD/$SESSION_ID.jsonl"

    # Build system message
    MSG="Session ID: $SESSION_ID
Transcript path: $TRANSCRIPT_PATH"

    # Add memory tool reminder on compaction
    if [[ "$TRIGGER" == "compact" ]]; then
        MSG="$MSG

Context was just compacted. Use the \`remember\` tool to retrieve relevant context from earlier in this conversation. Start with a broad query about what you were working on, then zoom into specific areas of interest."
    fi

    # Output JSON with additionalContext in the correct format
    jq -n --arg msg "$MSG" '{
      "hookSpecificOutput": {
        "hookEventName": "SessionStart",
        "additionalContext": $msg
      }
    }'
fi
