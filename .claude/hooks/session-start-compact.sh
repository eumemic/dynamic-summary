#!/bin/bash
# Instruct agent to use memory tool after compaction
#
# Input: JSON on stdin with session info
# Output: JSON with systemMessage to inject into context

cat << 'EOF'
{
  "systemMessage": "Context was just compacted. Use the `remember` tool to retrieve relevant context from earlier in this conversation. Start with a broad query about what you were working on, then zoom into specific areas of interest."
}
EOF
