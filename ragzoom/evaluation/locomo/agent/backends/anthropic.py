"""Claude Agent SDK backend implementing the BenchmarkingAgent protocol."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from typing import NamedTuple

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    ResultMessage,
    SdkMcpTool,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    create_sdk_mcp_server,
    query,
)

from ragzoom.evaluation.locomo.agent.protocol import (
    AgentResult,
    ToolDefinition,
    ToolResult,
    make_agent_result,
)
from ragzoom.model_info import ModelInfo

logger = logging.getLogger(__name__)


def _build_sdk_tool(
    td: ToolDefinition,
    retrieved_tokens: list[int],
) -> SdkMcpTool[dict[str, object]]:
    """Convert a ToolDefinition into an SdkMcpTool, tracking token counts."""
    param_types: dict[str, type] = {}
    for pname, pschema in td.parameters.items():
        if isinstance(pschema, dict):
            json_type = pschema.get("type", "string")
            if json_type == "integer":
                param_types[pname] = int
            elif json_type == "number":
                param_types[pname] = float
            elif json_type == "boolean":
                param_types[pname] = bool
            else:
                param_types[pname] = str
        else:
            param_types[pname] = str

    async def handler(args: dict[str, object]) -> dict[str, object]:
        tr: ToolResult = await td.handler(args)
        if tr.token_count > 0:
            retrieved_tokens.append(tr.token_count)
        if tr.is_error:
            return {
                "content": [{"type": "text", "text": tr.content}],
                "is_error": True,
            }
        return {"content": [{"type": "text", "text": tr.content}]}

    return SdkMcpTool(
        name=td.name,
        description=td.description,
        input_schema=param_types,
        handler=handler,
    )


class _UsageBreakdown(NamedTuple):
    """Detailed token usage from Anthropic's ResultMessage.

    Anthropic reports three categories of input tokens, each priced differently:
    - input_tokens: tokens after the last cache breakpoint (full input price)
    - cache_creation_tokens: newly written to cache (1.25x input price)
    - cache_read_tokens: served from cache (0.1x input price, 90% discount)
    """

    input_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
    output_tokens: int

    @property
    def total_input(self) -> int:
        """Total input tokens across all three categories."""
        return self.input_tokens + self.cache_creation_tokens + self.cache_read_tokens


def _extract_usage(message: ResultMessage) -> _UsageBreakdown:
    """Extract detailed token usage from a ResultMessage."""
    if message.usage is None:
        return _UsageBreakdown(0, 0, 0, 0)
    return _UsageBreakdown(
        input_tokens=int(message.usage.get("input_tokens", 0)),
        cache_creation_tokens=int(message.usage.get("cache_creation_input_tokens", 0)),
        cache_read_tokens=int(message.usage.get("cache_read_input_tokens", 0)),
        output_tokens=int(message.usage.get("output_tokens", 0)),
    )


def _compute_cost(model_id: str, usage: _UsageBreakdown) -> float | None:
    """Compute total cost in USD from usage breakdown and model pricing.

    Returns None if the model is not found in models.json.
    """
    try:
        info = ModelInfo()
        input_price, output_price = info.get_llm_costs(model_id)
        cache_discount = info.get_cache_discount(model_id)
        write_mult = info.get_cache_write_multiplier(model_id)
    except ValueError:
        logger.warning("Model %r not in models.json; cost not computed", model_id)
        return None

    input_cost = (usage.input_tokens / 1000) * input_price
    write_cost = (usage.cache_creation_tokens / 1000) * input_price * write_mult
    read_cost = (usage.cache_read_tokens / 1000) * input_price * (1 - cache_discount)
    output_cost = (usage.output_tokens / 1000) * output_price

    return input_cost + write_cost + read_cost + output_cost


class AnthropicBackend:
    """Claude backend for both agentic answers and single-shot judging.

    Uses ``query()`` for tool-free single-shot calls (judge path) and
    ``ClaudeSDKClient`` with MCP tools for agentic multi-turn answers.
    """

    def __init__(self, model_id: str) -> None:
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
    ) -> AgentResult:
        # jscpd:ignore-end
        """Generate a response, optionally using tools over multiple turns."""
        if temperature is not None:
            logger.warning(
                "AnthropicBackend: temperature=%.2f ignored (SDK does not support it)",
                temperature,
            )

        if not tools:
            return await self._generate_single_shot(system_prompt, user_prompt)
        return await self._generate_agentic(
            system_prompt, user_prompt, tools, max_turns
        )

    async def _generate_single_shot(
        self, system_prompt: str, user_prompt: str
    ) -> AgentResult:
        """Single-shot call via ``query()`` — used for judging."""
        start_time = time.monotonic()

        options = ClaudeAgentOptions(
            model=self._model_id,
            system_prompt=system_prompt,
            max_turns=1,
            permission_mode="bypassPermissions",
        )

        answer = ""
        usage = _UsageBreakdown(0, 0, 0, 0)

        async for message in query(prompt=user_prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        answer = block.text
            elif isinstance(message, ResultMessage):
                usage = _extract_usage(message)

        if not answer.strip():
            answer = "I don't know."

        return make_agent_result(
            answer=answer,
            total_input=usage.total_input,
            total_output=usage.output_tokens,
            retrieved_tokens=[],
            reasoning_turns=1,
            elapsed=time.monotonic() - start_time,
            total_cost_usd=_compute_cost(self._model_id, usage),
        )

    async def _generate_agentic(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: Sequence[ToolDefinition],
        max_turns: int,
    ) -> AgentResult:
        """Multi-turn agentic call via ``ClaudeSDKClient`` with MCP tools."""
        start_time = time.monotonic()
        retrieved_tokens: list[int] = []

        sdk_tools = [_build_sdk_tool(td, retrieved_tokens) for td in tools]

        memory_server = create_sdk_mcp_server(
            name="memory",
            version="1.0.0",
            tools=sdk_tools,
        )

        allowed_tools = [f"mcp__mem__{td.name}" for td in tools]

        options = ClaudeAgentOptions(
            model=self._model_id,
            system_prompt=system_prompt,
            mcp_servers={"mem": memory_server},
            allowed_tools=allowed_tools,
            max_turns=max_turns + 1,  # +1 for final answer turn
            permission_mode="bypassPermissions",
        )

        answer = ""
        usage = _UsageBreakdown(0, 0, 0, 0)
        reasoning_turns = 0

        async with ClaudeSDKClient(options=options) as client:
            await client.query(user_prompt)

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    reasoning_turns += 1
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            answer = block.text
                        elif isinstance(block, ToolUseBlock):
                            logger.debug(
                                "Tool call: %s(%s)",
                                block.name,
                                str(block.input)[:100],
                            )
                        elif isinstance(block, ToolResultBlock):
                            pass  # Logged in the tool handler

                elif isinstance(message, ResultMessage):
                    usage = _extract_usage(message)

        if not answer.strip():
            answer = "I don't know."

        return make_agent_result(
            answer=answer,
            total_input=usage.total_input,
            total_output=usage.output_tokens,
            retrieved_tokens=retrieved_tokens,
            reasoning_turns=reasoning_turns,
            elapsed=time.monotonic() - start_time,
            total_cost_usd=_compute_cost(self._model_id, usage),
        )
