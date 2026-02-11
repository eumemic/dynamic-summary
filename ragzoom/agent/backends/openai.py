"""OpenAI backend implementing the BenchmarkingAgent protocol."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import cast

from openai import AsyncOpenAI
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)

from ragzoom.agent.protocol import (
    AgentResult,
    AssistantTurn,
    MessageHistory,
    ToolCallRecord,
    ToolDefinition,
    ToolResultRecord,
    make_agent_result,
)
from ragzoom.cost import calculate_completion_cost, calculate_prompt_cost_with_cache
from ragzoom.model_info import ModelInfo

logger = logging.getLogger(__name__)


def _compute_cost(
    model_id: str,
    total_input: int,
    cached_input: int,
    total_output: int,
) -> float | None:
    """Compute total cost in USD using model pricing from models.json.

    Returns None if the model is not found in models.json.
    """
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


def _to_openai_tool(td: ToolDefinition) -> ChatCompletionToolParam:
    """Convert a ToolDefinition to OpenAI function-calling format."""
    return cast(
        ChatCompletionToolParam,
        {
            "type": "function",
            "function": {
                "name": td.name,
                "description": td.description,
                "parameters": {
                    "type": "object",
                    "properties": td.parameters,
                    "required": list(td.required),
                },
            },
        },
    )


def _history_to_openai_messages(
    history: MessageHistory,
) -> list[ChatCompletionMessageParam]:
    """Convert backend-agnostic MessageHistory to OpenAI message format."""
    messages: list[ChatCompletionMessageParam] = []
    for entry in history:
        if isinstance(entry, str):
            messages.append({"role": "user", "content": entry})
        elif isinstance(entry, AssistantTurn):
            if entry.tool_calls:
                tool_calls_payload = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.tool_name,
                            "arguments": tc.arguments_json,
                        },
                    }
                    for tc in entry.tool_calls
                ]
                msg: dict[str, object] = {
                    "role": "assistant",
                    "tool_calls": tool_calls_payload,
                }
                if entry.text is not None:
                    msg["content"] = entry.text
                messages.append(cast(ChatCompletionMessageParam, msg))
            else:
                messages.append({"role": "assistant", "content": entry.text or ""})
        elif isinstance(entry, ToolResultRecord):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry.call_id,
                    "content": entry.content,
                }
            )
    return messages


def _openai_messages_to_history(
    messages: list[ChatCompletionMessageParam],
) -> MessageHistory:
    """Convert OpenAI message list to backend-agnostic MessageHistory.

    Skips system messages — they are not part of the conversation history.
    """
    items: list[str | AssistantTurn | ToolResultRecord] = []
    for msg in messages:
        role = msg.get("role") if isinstance(msg, dict) else getattr(msg, "role", None)
        if role == "system":
            continue
        if role == "user":
            content = (
                msg.get("content", "")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            items.append(str(content) if content else "")
        elif role == "assistant":
            text: str | None = None
            tool_calls: list[ToolCallRecord] = []
            if isinstance(msg, dict):
                raw_content = msg.get("content")
                if raw_content is not None:
                    text = str(raw_content)
                raw_tcs = msg.get("tool_calls") or []
            else:
                raw_content = getattr(msg, "content", None)
                if raw_content is not None:
                    text = str(raw_content)
                raw_tcs = getattr(msg, "tool_calls", None) or []
            if isinstance(raw_tcs, list):
                for tc in raw_tcs:
                    if isinstance(tc, dict):
                        fn: dict[str, str] = tc.get("function", {})
                        tool_calls.append(
                            ToolCallRecord(
                                call_id=str(tc.get("id", "")),
                                tool_name=fn.get("name", ""),
                                arguments_json=fn.get("arguments", "{}"),
                            )
                        )
                    elif isinstance(tc, ChatCompletionMessageFunctionToolCall):
                        tool_calls.append(
                            ToolCallRecord(
                                call_id=tc.id,
                                tool_name=tc.function.name,
                                arguments_json=tc.function.arguments,
                            )
                        )
            items.append(AssistantTurn(text=text, tool_calls=tuple(tool_calls)))
        elif role == "tool":
            call_id = (
                msg.get("tool_call_id", "")
                if isinstance(msg, dict)
                else getattr(msg, "tool_call_id", "")
            )
            content = (
                msg.get("content", "")
                if isinstance(msg, dict)
                else getattr(msg, "content", "")
            )
            items.append(
                ToolResultRecord(
                    call_id=str(call_id),
                    content=str(content) if content else "",
                )
            )
    return tuple(items)


class OpenAIBackend:
    """OpenAI function-calling backend for both agentic answers and judging."""

    def __init__(self, client: AsyncOpenAI, model_id: str) -> None:
        self._client = client
        self._model_id = model_id

    # jscpd:ignore-start (BenchmarkingAgent protocol implementation)
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
        capture_history: bool = False,
        prior_history: MessageHistory | None = None,
    ) -> AgentResult:
        # jscpd:ignore-end
        """Generate a response, optionally using tools over multiple turns."""
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt},
        ]

        if prior_history is not None:
            messages.extend(_history_to_openai_messages(prior_history))
            capture_history = True

        messages.append({"role": "user", "content": user_prompt})

        oa_tools = [_to_openai_tool(td) for td in tools]
        tool_handlers = {td.name: td for td in tools}

        total_input = 0
        total_output = 0
        cached_input = 0
        reasoning_turns = 0
        retrieved_tokens: list[int] = []
        answer = ""
        start_time = time.monotonic()

        temp_arg: float | NotGiven | None = (
            float(temperature) if temperature is not None else NOT_GIVEN
        )

        response: ChatCompletion | None = None
        for _ in range(max_turns + 1):  # +1 for final answer turn
            reasoning_turns += 1
            calls_remaining = max_turns - len(retrieved_tokens)

            response = await self._client.chat.completions.create(
                model=self._model_id,
                messages=messages,
                tools=oa_tools if oa_tools and calls_remaining > 0 else [],
                temperature=temp_arg,
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

                    td = tool_handlers.get(tool_call.function.name)
                    if td is None:
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "content": f"Unknown tool: {tool_call.function.name}",
                            }
                        )
                        continue

                    args = json.loads(tool_call.function.arguments)
                    result = await td.handler(args)
                    if result.token_count > 0:
                        retrieved_tokens.append(result.token_count)

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": result.content,
                        }
                    )

                continue

            # Model returned a text answer
            answer = choice.message.content or ""
            break
        else:
            # Exhausted all turns without a final text answer
            assert response is not None
            last_content = response.choices[0].message.content
            answer = last_content if last_content else "I don't know."

        history: MessageHistory | None = None
        if capture_history:
            history = _openai_messages_to_history(messages)

        return make_agent_result(
            answer=answer,
            total_input=total_input,
            total_output=total_output,
            retrieved_tokens=retrieved_tokens,
            reasoning_turns=reasoning_turns,
            elapsed=time.monotonic() - start_time,
            total_cost_usd=_compute_cost(
                self._model_id, total_input, cached_input, total_output
            ),
            history=history,
        )
