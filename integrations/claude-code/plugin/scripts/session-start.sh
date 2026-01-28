#!/bin/bash
# Write session ID to PID-keyed temp file for MCP server lookup
# PPID is Claude Code's PID since this script is spawned directly by the hook runner

session_id=$(jq -r '.session_id // empty')
if [ -n "$session_id" ]; then
    echo "$session_id" > "/tmp/ragzoom-session-$PPID"
fi
