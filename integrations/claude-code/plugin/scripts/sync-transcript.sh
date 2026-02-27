#!/bin/bash
# Sync Claude Code transcript to RagZoom
#
# Input: JSON on stdin with transcript_path
# Output: Silent on success
#
# Runs sync in the background so the hook returns immediately.
# Uses a PID-based lockfile to skip concurrent syncs (next turn catches up).

set -euo pipefail

# Skip sync if disabled
if [[ "${RAGZOOM_DISABLE_SYNC:-}" == "1" ]]; then
    cat > /dev/null
    exit 0
fi

JSON=$(cat)
TRANSCRIPT_PATH=$(echo "$JSON" | jq -r '.transcript_path // ""')

if [[ -n "$TRANSCRIPT_PATH" && -f "$TRANSCRIPT_PATH" ]]; then
    LOCKFILE="/tmp/ragzoom-sync-$(basename "$TRANSCRIPT_PATH" .jsonl).pid"

    # Check if an existing sync is still running
    if [[ -f "$LOCKFILE" ]]; then
        EXISTING_PID=$(cat "$LOCKFILE" 2>/dev/null || echo "")
        if [[ -n "$EXISTING_PID" ]] && kill -0 "$EXISTING_PID" 2>/dev/null; then
            # Process is still alive — skip this sync
            exit 0
        fi
        # Process is dead — remove stale lockfile
        rm -f "$LOCKFILE"
    fi

    (
        echo "$BASHPID" > "$LOCKFILE" 2>/dev/null || exit 0
        trap 'rm -f "$LOCKFILE" 2>/dev/null || true' EXIT
        ragzoom-claude-code sync "$TRANSCRIPT_PATH" 2>/dev/null || true
    ) &
fi
