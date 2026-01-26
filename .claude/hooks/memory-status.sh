#!/bin/bash
# UserPromptSubmit hook to provide memory status to Claude

# Skip hooks when sync is disabled (e.g., RAGZOOM_DISABLE_SYNC=1 ralph build)
if [[ "${RAGZOOM_DISABLE_SYNC:-}" == "1" ]]; then
    cat > /dev/null  # consume stdin
    echo '{"additionalContext": ""}'
    exit 0
fi

# Get session ID from environment or input
input=$(cat)
session_id="${CLAUDE_SESSION_ID:-$(echo "$input" | jq -r '.session_id // empty')}"

if [ -z "$session_id" ]; then
    # No session ID, output empty JSON
    echo '{"additionalContext": ""}'
    exit 0
fi

# Query memory status using inline Python
status_text=$(.venv/bin/python3 -c "
import os
import sys
import time

STATE_FILE = os.path.expanduser('~/.claude/.memory-status-cache')
STALL_THRESHOLD_SECONDS = 60  # Consider stalled if no progress for 60s

def get_cached_state(document_id):
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) == 3 and parts[0] == document_id:
                        return int(parts[1]), float(parts[2])
    except (OSError, ValueError):
        pass
    return None

try:
    from ragzoom.client import GrpcRagzoomClient
    from ragzoom.daemon import ensure_server_running

    session_id = '$session_id'
    server_address = ensure_server_running()

    with GrpcRagzoomClient(server_address) as client:
        status = client.get_document_status(session_id)

    if not status.exists:
        print('')
        sys.exit(0)

    completion = status.completion_pct
    node_count = status.node_count
    complete_size = status.complete_forest_size
    leaf_count = status.leaf_count

    is_stalled = False
    incomplete = node_count < complete_size

    if incomplete:
        cached = get_cached_state(session_id)
        if cached is not None:
            prev_count, prev_time = cached
            elapsed = time.time() - prev_time
            if node_count == prev_count and elapsed > STALL_THRESHOLD_SECONDS:
                is_stalled = True
                stall_minutes = int(elapsed / 60)

    if is_stalled:
        print(f'⚠️ MEMORY STALLED: {completion:.0f}% complete ({node_count}/{complete_size} nodes), no progress for {stall_minutes}m. Daemon may need restart.')
    elif incomplete:
        print(f'Memory: {completion:.0f}% indexed ({node_count}/{complete_size} nodes, {leaf_count} leaves)')
    else:
        print(f'Memory: ✓ complete ({leaf_count} leaves)')
except Exception as e:
    print(f'Memory: unavailable ({e})')
" 2>/dev/null)

if [ -n "$status_text" ]; then
    # Escape for JSON
    escaped=$(echo "$status_text" | sed 's/"/\\"/g' | tr '\n' ' ')
    echo "{\"additionalContext\": \"$escaped\"}"
else
    echo '{"additionalContext": ""}'
fi
