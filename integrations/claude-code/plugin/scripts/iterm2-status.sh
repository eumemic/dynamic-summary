#!/bin/bash
# iTerm2 status bar script for ragzoom memory status
# Add as a "Shell Script" component in iTerm2 status bar with 2-3 second refresh
#
# Scans existing /tmp/ragzoom-session-* files (no wrapper needed)

session_id=""

# Find the most recently modified ragzoom session file with a live process
for session_file in $(ls -t /tmp/ragzoom-session-* 2>/dev/null); do
    # Extract PID from filename
    pid="${session_file##*-}"

    # Skip non-numeric (e.g., ragzoom-session-tty directory if it exists)
    [[ ! "$pid" =~ ^[0-9]+$ ]] && continue

    if kill -0 "$pid" 2>/dev/null; then
        session_id=$(cat "$session_file")
        break
    else
        # Stale file, clean up
        rm -f "$session_file"
    fi
done

if [[ -z "$session_id" ]]; then
    echo ""
    exit 0
fi

# Query ragzoom for memory status
status_json=$(ragzoom document-status "$session_id" --json --server-address localhost:50051 2>/dev/null)

if [[ -z "$status_json" ]]; then
    echo ""
    exit 0
fi

exists=$(echo "$status_json" | jq -r '.exists')
if [[ "$exists" != "true" ]]; then
    echo ""
    exit 0
fi

node_count=$(echo "$status_json" | jq -r '.node_count')
complete_size=$(echo "$status_json" | jq -r '.complete_forest_size')
pending=$((complete_size - node_count))

if [[ "$pending" -gt 0 ]]; then
    echo "🧠 indexing $pending"
else
    echo "🧠 ready"
fi
