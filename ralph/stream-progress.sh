#!/bin/bash
# ralph/stream-progress.sh - Format Claude stream-json output for humans
#
# Reads JSONL from stdin and outputs human-readable progress:
# - Full assistant text responses
# - Tool calls as ToolName(truncated_args...)
# - Tool results are hidden (too noisy)

set -euo pipefail

while IFS= read -r line; do
    # Skip empty lines
    [ -z "$line" ] && continue

    type=$(echo "$line" | jq -r '.type // empty' 2>/dev/null) || continue

    # Skip subagent messages (have parent_tool_use_id)
    parent_id=$(echo "$line" | jq -r '.parent_tool_use_id // empty' 2>/dev/null) || true
    [ -n "$parent_id" ] && continue

    case "$type" in
        assistant)
            # Print text content (full assistant messages)
            text=$(echo "$line" | jq -r '
                [.message.content[]? | select(.type=="text") | .text] | join("")
            ' 2>/dev/null) || true

            if [ -n "$text" ] && [ "$text" != "null" ]; then
                echo "$text"
                echo "---"
            fi

            # Print tool calls as ToolName(args...)
            # Truncate at first newline if present
            tools=$(echo "$line" | jq -r '
                .message.content[]? |
                select(.type=="tool_use") |
                .name as $name |
                (.input | tostring) as $args |
                ($args | split("\\n")[0]) as $first_line |
                if $first_line != $args then
                    "\($name)(\($first_line)...)"
                else
                    "\($name)(\($args))"
                end
            ' 2>/dev/null) || true

            if [ -n "$tools" ]; then
                echo "$tools"
                echo "---"
            fi
            ;;

        # Skip: user (tool results), system, file-history-snapshot, summary
    esac
done
