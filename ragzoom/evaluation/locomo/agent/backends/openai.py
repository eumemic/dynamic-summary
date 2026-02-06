"""OpenAI function-calling agent that iteratively zooms via recall."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from ragzoom.evaluation.locomo.agent.prompt import (
    AGENT_SYSTEM_PROMPT,
    RECALL_TOOL_SCHEMA,
)
from ragzoom.evaluation.locomo.agent.protocol import AgentResult
from ragzoom.evaluation.locomo.types import CostMetrics
from ragzoom.wrapper import RagZoom

logger = logging.getLogger(__name__)


class OpenAIAgentBackend:
    """OpenAI function-calling agent that iteratively zooms via recall."""

    def __init__(
        self,
        client: AsyncOpenAI,
        rz: RagZoom,
        model_id: str,
    ) -> None:
        self._client = client
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
        tool: ChatCompletionToolParam = cast(
            ChatCompletionToolParam, RECALL_TOOL_SCHEMA
        )

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"You have {max_iterations} recall calls available. "
                    f"Default budget per call: {budget_tokens} tokens."
                ),
            },
        ]

        total_input = 0
        total_output = 0
        reasoning_turns = 0
        retrieved_tokens: list[int] = []
        answer = ""
        start_time = time.monotonic()

        for _ in range(max_iterations + 1):  # +1 for final answer turn
            reasoning_turns += 1
            calls_remaining = max_iterations - len(retrieved_tokens)

            response = await self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                tools=[tool] if calls_remaining > 0 else [],
                temperature=0.0,
            )

            if response.usage:
                total_input += response.usage.prompt_tokens
                total_output += response.usage.completion_tokens

            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(cast(ChatCompletionMessageParam, choice.message))

                for tool_call in choice.message.tool_calls:
                    if not isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                        continue

                    if tool_call.function.name != "recall":
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Unknown tool: {tool_call.function.name}",
                            }
                        )
                        continue

                    args = json.loads(tool_call.function.arguments)
                    query_text = str(args["query"])
                    call_budget = int(args.get("budget_tokens", budget_tokens))
                    time_start: str | None = args.get("time_start")
                    time_end: str | None = args.get("time_end")

                    try:
                        query_response = await asyncio.to_thread(
                            self._rz.query,
                            doc_id,
                            query_text,
                            budget_tokens=call_budget,
                            time_start=time_start,
                            time_end=time_end,
                        )
                    except Exception as exc:
                        logger.warning("recall(%s) failed: %s", query_text[:50], exc)
                        retrieved_tokens.append(0)
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Error: {exc}",
                            }
                        )
                        continue

                    retrieved_tokens.append(query_response.token_count)
                    logger.debug(
                        "recall(%s, budget=%d) → %d tokens",
                        query_text[:50],
                        call_budget,
                        query_response.token_count,
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": query_response.summary,
                        }
                    )

                continue

            # Model returned a text answer
            answer = choice.message.content or ""
            break
        else:
            # Exhausted all iterations without a final text answer
            if choice.message.content:
                answer = choice.message.content
            else:
                answer = "I don't know."

        # jscpd:ignore-start (same structure as AnthropicAgentBackend)
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
        # jscpd:ignore-end
