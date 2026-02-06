"""System prompt and tool schema for the agentic LoCoMo evaluator."""

from __future__ import annotations

AGENT_SYSTEM_PROMPT = """\
You are answering questions about a conversation between two people.
You have access to a memory retrieval tool called `recall` that returns \
summarized context from the conversation at various levels of detail.

## Strategy: Iterative Zoom
1. SURVEY: Start with a broad query using your full token budget to get an overview.
2. IDENTIFY: Look for time ranges or topics in the returned summaries that \
seem relevant to the question.
3. ZOOM: Call `recall` again with a tighter time window and/or different \
query to get more detailed (verbatim) content.
4. REPEAT: Keep drilling until you have enough verbatim detail to answer \
confidently.

## Rules
- Use ALL your retrieval calls wisely. Start broad, then narrow.
- When you have enough information, respond with your final answer.
- If the context does not contain enough information after zooming, say \
"I don't know."
- Give the most information-dense answer possible that fully answers the \
question.
- Avoid filler words, hedging, or restating the question.
- Your final answer should be ONLY the answer text, no explanation of your \
process."""


RECALL_TOOL_SCHEMA: dict[str, object] = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Retrieve summarized context from the conversation. "
            "Use budget_tokens to control detail level (higher = more content). "
            "Use time_start/time_end (ISO 8601) to zoom into specific time ranges."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query to find relevant content",
                },
                "budget_tokens": {
                    "type": "integer",
                    "description": (
                        "Maximum tokens in response " "(higher = more detail)"
                    ),
                },
                "time_start": {
                    "type": "string",
                    "description": ("ISO 8601 timestamp to start from (optional)"),
                },
                "time_end": {
                    "type": "string",
                    "description": ("ISO 8601 timestamp to end at (optional)"),
                },
            },
            "required": ["query", "budget_tokens"],
        },
    },
}

# Anthropic Messages API tool format (same semantics, different schema layout)
RECALL_TOOL_SCHEMA_ANTHROPIC: dict[str, object] = {
    "name": "recall",
    "description": (
        "Retrieve summarized context from the conversation. "
        "Use budget_tokens to control detail level (higher = more content). "
        "Use time_start/time_end (ISO 8601) to zoom into specific time ranges."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query to find relevant content",
            },
            "budget_tokens": {
                "type": "integer",
                "description": "Maximum tokens in response (higher = more detail)",
            },
            "time_start": {
                "type": "string",
                "description": "ISO 8601 timestamp to start from (optional)",
            },
            "time_end": {
                "type": "string",
                "description": "ISO 8601 timestamp to end at (optional)",
            },
        },
        "required": ["query", "budget_tokens"],
    },
}
