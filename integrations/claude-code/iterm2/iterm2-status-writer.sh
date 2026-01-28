#!/bin/bash
# Writes ragzoom status per-PID for iTerm2 to read
# Run in background: nohup ~/.claude/plugins/ragzoom-memory/scripts/iterm2-status-writer.sh &

STATUS_DIR="/tmp/ragzoom-status"
mkdir -p "$STATUS_DIR"

while true; do
    # Find all live Claude processes with session files
    for session_file in /tmp/ragzoom-session-*; do
        [[ ! -f "$session_file" ]] && continue

        pid="${session_file##*-}"
        [[ ! "$pid" =~ ^[0-9]+$ ]] && continue

        # Check if process is alive
        kill -0 "$pid" 2>/dev/null || continue

        session_id=$(cat "$session_file")
        [[ -z "$session_id" ]] && continue

        # Query ragzoom
        json=$(ragzoom document-status "$session_id" --json --server-address localhost:50051 2>/dev/null)
        exists=$(echo "$json" | jq -r '.exists' 2>/dev/null)

        status_file="$STATUS_DIR/$pid"

        if [[ "$exists" == "true" ]]; then
            node_count=$(echo "$json" | jq -r '.node_count')
            complete_size=$(echo "$json" | jq -r '.complete_forest_size')
            pending=$((complete_size - node_count))

            if [[ "$pending" -gt 0 ]]; then
                echo "memory: building $pending summaries" > "$status_file"
            else
                echo "memory: synced" > "$status_file"
            fi
        else
            echo "" > "$status_file"
        fi
    done

    sleep 2
done
