#!/bin/bash
# Sync Claude Code transcript to RagZoom on session stop
#
# Input: JSON on stdin with transcript_path
# Output: Silent on success

set -euo pipefail

# Skip sync if disabled
if [[ "${RAGZOOM_DISABLE_SYNC:-}" == "1" ]]; then
    cat > /dev/null
    exit 0
fi

JSON=$(cat)
TRANSCRIPT_PATH=$(echo "$JSON" | jq -r '.transcript_path // ""')

if [[ -n "$TRANSCRIPT_PATH" && -f "$TRANSCRIPT_PATH" ]]; then
    ragzoom-claude-code sync "$TRANSCRIPT_PATH"
fi
