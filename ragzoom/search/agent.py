"""Agentic search: iterative zoom loop driven by an LLM."""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, cast

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from ragzoom.cost import calculate_completion_cost, calculate_prompt_cost_with_cache
from ragzoom.model_info import ModelInfo
from ragzoom.output_formatters import format_tiling_spans
from ragzoom.search.config import SearchConfig
from ragzoom.search.prompt import SEARCH_SYSTEM_PROMPT
from ragzoom.search.types import SearchIteration, SearchProfile, SearchResult
from ragzoom.server.query_executor import execute_query_internal

if TYPE_CHECKING:
    from ragzoom.server.state import ServerState

logger = logging.getLogger(__name__)

RECALL_TOOL: ChatCompletionToolParam = cast(
    ChatCompletionToolParam,
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Retrieve summarized context from the conversation. "
                "Returns variable-resolution text with Span tags showing "
                "time ranges and summarization height."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Semantic search keywords",
                    },
                    "budget_tokens": {
                        "type": "integer",
                        "description": (
                            "Max tokens in response (higher = more detail, "
                            "max: {max_budget})"
                        ),
                    },
                    "time_start": {
                        "type": "string",
                        "description": "ISO 8601 lower bound (optional)",
                    },
                    "time_end": {
                        "type": "string",
                        "description": "ISO 8601 upper bound (optional)",
                    },
                },
                "required": ["query", "budget_tokens"],
            },
        },
    },
)


def _build_recall_tool(max_budget: int) -> ChatCompletionToolParam:
    """Build the recall tool definition with the actual max budget."""
    tool = json.loads(json.dumps(RECALL_TOOL))
    desc = tool["function"]["parameters"]["properties"]["budget_tokens"]["description"]
    tool["function"]["parameters"]["properties"]["budget_tokens"]["description"] = (
        desc.replace("{max_budget}", str(max_budget))
    )
    return cast(ChatCompletionToolParam, tool)


def _compute_cost(
    model_id: str,
    total_input: int,
    cached_input: int,
    total_output: int,
) -> float | None:
    """Compute total cost in USD from model pricing."""
    try:
        info = ModelInfo()
        input_price, output_price = info.get_llm_costs(model_id)
        cache_discount = info.get_cache_discount(model_id)
    except ValueError:
        logger.warning("Model %r not in models.json; cost not computed", model_id)
        return None

    prompt_cost = calculate_prompt_cost_with_cache(
        total_input, cached_input, input_price, cache_discount
    )
    output_cost = calculate_completion_cost(total_output, output_price)
    return prompt_cost + output_cost


class SearchAgent:
    """LLM-driven iterative zoom agent for answering questions from memory."""

    def __init__(self, config: SearchConfig, openai_client: AsyncOpenAI) -> None:
        self._config = config
        self._client = openai_client

    async def search(
        self,
        question: str,
        document_id: str,
        state: ServerState,
    ) -> SearchResult:
        """Run the agentic search loop.

        Args:
            question: The user's question to answer.
            document_id: Document to search within.
            state: Server state for in-process query execution.

        Returns:
            SearchResult with the answer and optional profiling data.
        """
        config = self._config
        profiling = config.profiling_enabled
        start_time = time.monotonic()

        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        recall_tool = _build_recall_tool(config.max_token_budget)
        iterations: list[SearchIteration] = []
        total_input = 0
        total_output = 0
        cached_input = 0
        answer = ""

        response: ChatCompletion | None = None
        max_rounds = config.max_iterations + 1  # +1 for final answer turn

        for _ in range(max_rounds):
            calls_remaining = config.max_iterations - len(iterations)

            response = await self._client.chat.completions.create(
                model=config.agent_model,
                messages=messages,
                tools=[recall_tool] if calls_remaining > 0 else [],
            )

            if response.usage:
                total_input += response.usage.prompt_tokens
                total_output += response.usage.completion_tokens
                details = response.usage.prompt_tokens_details
                if details is not None and details.cached_tokens is not None:
                    cached_input += details.cached_tokens

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
                    result_text, result_tokens = await self._execute_recall(
                        args, document_id, state
                    )

                    # Record reasoning from the assistant message content
                    agent_reasoning = choice.message.content or ""

                    if profiling:
                        iterations.append(
                            SearchIteration(
                                query=str(args.get("query", "")),
                                budget_tokens=int(
                                    str(
                                        args.get(
                                            "budget_tokens", config.max_token_budget
                                        )
                                    )
                                ),
                                time_start=_str_or_none(args.get("time_start")),
                                time_end=_str_or_none(args.get("time_end")),
                                result_text=result_text,
                                result_token_count=result_tokens,
                                agent_reasoning=agent_reasoning,
                            )
                        )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result_text,
                        }
                    )
                continue

            # Model returned a text answer
            answer = choice.message.content or ""
            break
        else:
            # Exhausted all rounds without a final text answer
            if response is not None:
                last_content = response.choices[0].message.content
                answer = last_content if last_content else "I don't know."

        elapsed = time.monotonic() - start_time

        profile: SearchProfile | None = None
        if profiling:
            transcript = _format_transcript(messages)
            retrospective = await self._run_retrospective(question, answer, transcript)
            profile = SearchProfile(
                iterations=tuple(iterations),
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                total_cost_usd=_compute_cost(
                    config.agent_model, total_input, cached_input, total_output
                ),
                duration_seconds=elapsed,
                retrospective=retrospective,
                transcript=transcript,
            )

        return SearchResult(answer=answer, profile=profile)

    async def _execute_recall(
        self,
        args: dict[str, object],
        document_id: str,
        state: ServerState,
    ) -> tuple[str, int]:
        """Execute a recall tool call, returning (formatted_text, token_count)."""
        query_text = str(args.get("query", ""))
        raw_budget = args.get("budget_tokens")
        budget = (
            min(int(str(raw_budget)), self._config.max_token_budget)
            if raw_budget is not None
            else self._config.max_token_budget
        )
        raw_start = args.get("time_start")
        time_start = str(raw_start) if raw_start and raw_start != "" else None
        raw_end = args.get("time_end")
        time_end = str(raw_end) if raw_end and raw_end != "" else None

        try:
            query_output = await execute_query_internal(
                state,
                document_id=document_id,
                query=query_text,
                budget_tokens=budget,
                time_start=time_start,
                time_end=time_end,
            )
            formatted = format_tiling_spans(query_output)
            token_count = query_output.query_result.token_count
            logger.debug(
                "recall(%s, budget=%d) → %d tokens",
                query_text[:50],
                budget,
                token_count,
            )
            return formatted, token_count
        except Exception as exc:
            logger.warning("recall(%s) failed: %s", query_text[:50], exc)
            return f"Error: {exc}", 0

    async def _run_retrospective(
        self, question: str, answer: str, transcript: str
    ) -> str:
        """Run a retrospective critique of the search process."""
        from ragzoom.search.retrospective import run_retrospective

        return await run_retrospective(
            self._client,
            self._config.agent_model,
            question=question,
            answer=answer,
            transcript=transcript,
        )


def _str_or_none(val: object) -> str | None:
    """Convert a value to str or None."""
    if val is None or val == "":
        return None
    return str(val)


def _format_transcript(messages: list[ChatCompletionMessageParam]) -> str:
    """Format the full LLM message trace as readable text."""
    lines: list[str] = []
    for msg in messages:
        raw = cast(dict[str, object], msg)
        role = str(raw.get("role", "unknown"))
        content = raw.get("content", "")
        if role == "system":
            lines.append(f"[SYSTEM] {str(content)[:200]}...")
        elif role == "user":
            lines.append(f"[USER] {content}")
        elif role == "assistant":
            text = str(content) if content else ""
            tool_calls = raw.get("tool_calls")
            if tool_calls is not None and isinstance(tool_calls, list):
                for tc in tool_calls:
                    fn = getattr(tc, "function", None)
                    if fn:
                        lines.append(
                            f"[ASSISTANT] tool_call: {fn.name}({fn.arguments})"
                        )
            if text:
                lines.append(f"[ASSISTANT] {text}")
        elif role == "tool":
            tool_content = str(content)
            if len(tool_content) > 500:
                tool_content = tool_content[:500] + "..."
            lines.append(f"[TOOL] {tool_content}")
    return "\n".join(lines)
