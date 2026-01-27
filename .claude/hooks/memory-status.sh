#!/bin/bash
# UserPromptSubmit hook - provides memory status to Claude
# Uses shared script at ~/.claude/memory-status.sh for consistent output with statusline

input=$(cat)
session_id=$(echo "$input" | jq -r '.session_id // empty')

if [ -z "$session_id" ] || [ ! -x ~/.claude/memory-status.sh ]; then
    exit 0
fi

status=$(~/.claude/memory-status.sh "$session_id")

if [ -n "$status" ]; then
    escaped=$(echo "$status" | sed 's/"/\\"/g')
    echo "{\"hookSpecificOutput\": {\"hookEventName\": \"UserPromptSubmit\", \"additionalContext\": \"$escaped\"}}"
fi
