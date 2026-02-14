#!/bin/bash
# Sync Claude Code transcript to RagZoom
#
# Input: JSON on stdin with transcript_path
# Output: Silent on success
#
# Runs sync in the background so the hook returns immediately.
# Uses a lockdir to skip concurrent syncs (next turn catches up).

set -euo pipefail

# Skip sync if disabled
if [[ "${RAGZOOM_DISABLE_SYNC:-}" == "1" ]]; then
    cat > /dev/null
    exit 0
fi

JSON=$(cat)
TRANSCRIPT_PATH=$(echo "$JSON" | jq -r '.transcript_path // ""')

if [[ -n "$TRANSCRIPT_PATH" && -f "$TRANSCRIPT_PATH" ]]; then
    LOCKDIR="/tmp/ragzoom-sync-$(basename "$TRANSCRIPT_PATH" .jsonl).lock"

    # Remove stale lock (> 2 min old implies crashed sync)
    if [[ -d "$LOCKDIR" ]]; then
        find "$LOCKDIR" -maxdepth 0 -mmin +2 -exec rm -rf {} + 2>/dev/null || true
    fi

    (
        mkdir "$LOCKDIR" 2>/dev/null || exit 0
        trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT
        ragzoom-claude-code sync "$TRANSCRIPT_PATH" 2>/dev/null || true
    ) &
fi
