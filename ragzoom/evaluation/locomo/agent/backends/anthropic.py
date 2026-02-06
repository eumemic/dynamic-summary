"""Claude Agent SDK backend that iteratively zooms via recall."""

from __future__ import annotations

import logging
import time

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    tool,
)

from ragzoom.evaluation.locomo.agent.prompt import AGENT_SYSTEM_PROMPT
from ragzoom.evaluation.locomo.agent.protocol import AgentResult
from ragzoom.evaluation.locomo.types import CostMetrics
from ragzoom.wrapper import RagZoom

logger = logging.getLogger(__name__)


class AnthropicAgentBackend:
    """Claude Agent SDK backend that iteratively zooms via recall.

    Uses the official Claude Agent SDK with a custom MCP tool for memory
    retrieval. The SDK handles the agentic loop, tool execution, and
    authentication (including OAuth tokens) automatically.
    """

    def __init__(self, rz: RagZoom, model_id: str) -> None:
        self._rz = rz
        self._model_id = model_id

    async def generate(
        self,
        doc_id: str,
        question: str,
        budget_tokens: int,
        max_iterations: int,
    ) -> AgentResult:
        """Run the agentic zoom loop to answer a question.

        The agent gets up to *max_iterations* recall tool calls. After
        exhausting its calls (or choosing to stop early), it must produce
        a text answer.
        """
        start_time = time.monotonic()
        retrieved_tokens: list[int] = []
        retrieval_call_count = 0

        # Create the recall tool that captures doc_id and budget_tokens
        @tool(
            "recall",
            "Retrieve summarized context from the conversation. "
            "Use budget_tokens to control detail level (higher = more content). "
            "Use time_start/time_end (ISO 8601) to zoom into specific time ranges.",
            {
                "query": str,
                "budget_tokens": int,
                "time_start": str,
                "time_end": str,
            },
        )
        async def recall_tool(args: dict[str, object]) -> dict[str, object]:
            nonlocal retrieval_call_count
            retrieval_call_count += 1

            query_text = str(args.get("query", ""))
            raw_budget = args.get("budget_tokens")
            call_budget = (
                int(str(raw_budget)) if raw_budget is not None else budget_tokens
            )

            # Parse time bounds, converting empty strings to None
            raw_start = args.get("time_start")
            time_start: str | None = (
                str(raw_start) if raw_start and raw_start != "" else None
            )
            raw_end = args.get("time_end")
            time_end: str | None = str(raw_end) if raw_end and raw_end != "" else None

            try:
                # RagZoom query is synchronous - wrap in executor would be
                # ideal but the SDK runs in-process so blocking is acceptable
                query_response = self._rz.query(
                    doc_id,
                    query_text,
                    budget_tokens=call_budget,
                    time_start=time_start,
                    time_end=time_end,
                )
                retrieved_tokens.append(query_response.token_count)
                logger.debug(
                    "recall(%s, budget=%d) → %d tokens",
                    query_text[:50],
                    call_budget,
                    query_response.token_count,
                )
                return {"content": [{"type": "text", "text": query_response.summary}]}
            except Exception as exc:
                logger.warning("recall(%s) failed: %s", query_text[:50], exc)
                retrieved_tokens.append(0)
                return {
                    "content": [{"type": "text", "text": f"Error: {exc}"}],
                    "is_error": True,
                }

        # Create SDK MCP server with the recall tool
        memory_server = create_sdk_mcp_server(
            name="memory",
            version="1.0.0",
            tools=[recall_tool],
        )

        # Configure agent options
        options = ClaudeAgentOptions(
            model=self._model_id,
            system_prompt=AGENT_SYSTEM_PROMPT,
            mcp_servers={"mem": memory_server},
            allowed_tools=["mcp__mem__recall"],
            max_turns=max_iterations + 1,  # +1 for final answer turn
            permission_mode="bypassPermissions",  # No user prompts during benchmark
        )

        # Run the agent
        answer = ""
        total_input = 0
        total_output = 0
        reasoning_turns = 0

        async with ClaudeSDKClient(options=options) as client:
            prompt = (
                f"Question: {question}\n\n"
                f"You have {max_iterations} recall calls available. "
                f"Default budget per call: {budget_tokens} tokens."
            )
            await client.query(prompt)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    reasoning_turns += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            # Capture the last text block as the answer
                            answer = block.text
                        elif isinstance(block, ToolUseBlock):
                            logger.debug(
                                "Tool call: %s(%s)",
                                block.name,
                                str(block.input)[:100],
                            )
                        elif isinstance(block, ToolResultBlock):
                            pass  # Tool results are logged in the tool itself

                elif isinstance(message, ResultMessage):
                    # Extract usage metrics from the result
                    if message.usage is not None:
                        usage = message.usage
                        total_input = int(usage.get("input_tokens", 0))
                        total_output = int(usage.get("output_tokens", 0))

        # If no answer was captured, use a fallback
        if not answer.strip():
            answer = "I don't know."

        return AgentResult(
            answer=answer,
            cost=CostMetrics(
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                retrieval_call_count=len(retrieved_tokens),
                reasoning_turn_count=reasoning_turns,
                retrieved_tokens_per_call=tuple(retrieved_tokens),
                query_duration_seconds=time.monotonic() - start_time,
            ),
        )
