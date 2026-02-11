"""Claude Agent SDK backend implementing the BenchmarkingAgent protocol."""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
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

from ragzoom.agent.protocol import (
    AgentResult,
    AssistantTurn,
    MessageHistory,
    ToolDefinition,
    ToolResult,
    ToolResultRecord,
    make_agent_result,
)
from ragzoom.model_info import ModelInfo

logger = logging.getLogger(__name__)

# Bump the SDK's initialize timeout from the default 60s to 300s.  Claude Max
# can be slow to respond under load.  The env var is in milliseconds; the SDK
# uses max(value/1000, 60).
os.environ.setdefault("CLAUDE_CODE_STREAM_CLOSE_TIMEOUT", "300000")

_SDK_MAX_RETRIES = 2
_SDK_RETRY_BACKOFF_BASE = 5.0  # seconds; retries at 5s, 10s


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


def _format_history_as_context(history: MessageHistory) -> str:
    """Format prior conversation turns as readable text for the system prompt.

    The Anthropic SDK manages its own internal message list, so we can't inject
    native messages. Instead, we append a formatted transcript to the system
    prompt so the model has full context from prior turns.
    """
    lines: list[str] = ["## Prior conversation turns"]
    for entry in history:
        if isinstance(entry, str):
            lines.append(f"\n[User]\n{entry}")
        elif isinstance(entry, AssistantTurn):
            if entry.text:
                lines.append(f"\n[Assistant]\n{entry.text}")
            for tc in entry.tool_calls:
                lines.append(f"\n[Tool call: {tc.tool_name}({tc.arguments_json})]")
        elif isinstance(entry, ToolResultRecord):
            prefix = "[Tool error]" if entry.is_error else "[Tool result]"
            # Truncate long tool results to keep system prompt manageable
            content = entry.content
            if len(content) > 500:
                content = content[:500] + "..."
            lines.append(f"\n{prefix}\n{content}")
    return "\n".join(lines)


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
        capture_history: bool = False,
        prior_history: MessageHistory | None = None,
    ) -> AgentResult:
        # jscpd:ignore-end
        """Generate a response, optionally using tools over multiple turns."""
        if temperature is not None:
            logger.warning(
                "AnthropicBackend: temperature=%.2f ignored (SDK does not support it)",
                temperature,
            )

        if prior_history is not None:
            system_prompt = (
                system_prompt + "\n\n" + _format_history_as_context(prior_history)
            )
            capture_history = True

        if not tools:
            result = await self._generate_single_shot(system_prompt, user_prompt)
        else:
            result = await self._generate_agentic(
                system_prompt, user_prompt, tools, max_turns
            )

        if not capture_history:
            return result

        # Build minimal history: prior turns + this turn
        new_turn: tuple[str | AssistantTurn | ToolResultRecord, ...] = (
            user_prompt,
            AssistantTurn(text=result.answer),
        )
        if prior_history is not None:
            combined = prior_history + new_turn
        else:
            combined = new_turn
        return AgentResult(answer=result.answer, cost=result.cost, history=combined)

    async def _generate_single_shot(
        self, system_prompt: str, user_prompt: str
    ) -> AgentResult:
        """Single-shot call via ``query()`` — used for judging."""
        start_time = time.monotonic()

        with tempfile.TemporaryDirectory(prefix="claude_sdk_") as tmpdir:
            options = ClaudeAgentOptions(
                model=self._model_id,
                system_prompt=system_prompt,
                max_turns=1,
                permission_mode="bypassPermissions",
                env={"XDG_DATA_HOME": tmpdir},
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
        """Multi-turn agentic call via ``ClaudeSDKClient`` with MCP tools.

        Each subprocess gets an isolated ``XDG_DATA_HOME`` so the Claude CLI's
        PID lock (on ``$XDG_DATA_HOME/claude/versions/<ver>``) doesn't collide
        across concurrent clients.
        """
        start_time = time.monotonic()
        retrieved_tokens: list[int] = []

        sdk_tools = [_build_sdk_tool(td, retrieved_tokens) for td in tools]

        memory_server = create_sdk_mcp_server(
            name="memory",
            version="1.0.0",
            tools=sdk_tools,
        )

        allowed_tools = [f"mcp__mem__{td.name}" for td in tools]

        answer = ""
        usage = _UsageBreakdown(0, 0, 0, 0)
        reasoning_turns = 0
        last_err: Exception | None = None
        for attempt in range(_SDK_MAX_RETRIES + 1):
            if attempt > 0:
                backoff = _SDK_RETRY_BACKOFF_BASE * (2 ** (attempt - 1))
                logger.warning(
                    "SDK init retry %d/%d after %.0fs backoff",
                    attempt,
                    _SDK_MAX_RETRIES,
                    backoff,
                )
                await asyncio.sleep(backoff)

            with tempfile.TemporaryDirectory(prefix="claude_sdk_") as tmpdir:
                options = ClaudeAgentOptions(
                    model=self._model_id,
                    system_prompt=system_prompt,
                    mcp_servers={"mem": memory_server},
                    allowed_tools=allowed_tools,
                    max_turns=max_turns + 1,  # +1 for final answer turn
                    permission_mode="bypassPermissions",
                    env={"XDG_DATA_HOME": tmpdir},
                )

                try:
                    async with ClaudeSDKClient(options=options) as client:
                        answer, usage, reasoning_turns = (
                            await self._run_sdk_conversation(client, user_prompt)
                        )
                    break
                except Exception as exc:
                    last_err = exc
                    if attempt < _SDK_MAX_RETRIES:
                        logger.warning("SDK attempt %d failed: %s", attempt + 1, exc)
                        continue
                    raise RuntimeError(
                        f"ClaudeSDKClient failed after {_SDK_MAX_RETRIES + 1} attempts"
                    ) from last_err

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

    @staticmethod
    async def _run_sdk_conversation(
        client: ClaudeSDKClient, user_prompt: str
    ) -> tuple[str, _UsageBreakdown, int]:
        """Drive the SDK conversation after initialization."""
        await client.query(user_prompt)

        answer = ""
        usage = _UsageBreakdown(0, 0, 0, 0)
        reasoning_turns = 0

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

        return answer, usage, reasoning_turns
