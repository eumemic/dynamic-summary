"""Claude Agent SDK backend implementing the BenchmarkingAgent protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
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
    ToolCallRecord,
    ToolDefinition,
    ToolResult,
    ToolResultRecord,
    make_agent_result,
)
from ragzoom.daemon import get_daemon_state_dir
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


# ---------------------------------------------------------------------------
# SDK conversation result with history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ConversationResult:
    """Full result from an SDK conversation, including history."""

    answer: str
    usage: _UsageBreakdown
    reasoning_turns: int
    sdk_session_id: str | None
    history: MessageHistory


def _build_history_from_stream(
    user_prompt: str,
    messages: list[AssistantMessage],
) -> MessageHistory:
    """Build a MessageHistory from the user prompt and SDK assistant messages.

    Each AssistantMessage may contain TextBlock, ToolUseBlock, and ToolResultBlock
    items. We group consecutive text+tool_use blocks into AssistantTurn objects,
    and emit ToolResultRecord entries for tool results.
    """
    items: list[str | AssistantTurn | ToolResultRecord] = [user_prompt]

    for msg in messages:
        text_parts: list[str] = []
        tool_calls: list[ToolCallRecord] = []

        for block in msg.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
            elif isinstance(block, ToolUseBlock):
                tool_calls.append(
                    ToolCallRecord(
                        call_id=block.id,
                        tool_name=block.name,
                        arguments_json=json.dumps(block.input),
                    )
                )
            elif isinstance(block, ToolResultBlock):
                # Flush any pending assistant turn before emitting a tool result
                if text_parts or tool_calls:
                    items.append(
                        AssistantTurn(
                            text="\n".join(text_parts) if text_parts else None,
                            tool_calls=tuple(tool_calls),
                        )
                    )
                    text_parts = []
                    tool_calls = []

                content = block.content if isinstance(block.content, str) else ""
                items.append(
                    ToolResultRecord(
                        call_id=block.tool_use_id,
                        content=content,
                    )
                )

        # Flush remaining assistant content after all blocks in this message
        if text_parts or tool_calls:
            items.append(
                AssistantTurn(
                    text="\n".join(text_parts) if text_parts else None,
                    tool_calls=tuple(tool_calls),
                )
            )

    return tuple(items)


class AnthropicBackend:
    """Claude backend for both agentic answers and single-shot judging.

    Uses ``query()`` for tool-free single-shot calls (judge path) and
    ``ClaudeSDKClient`` with MCP tools for agentic multi-turn answers.

    Agentic sessions use persistent directories under the daemon state dir,
    enabling SDK-native resume via ``ClaudeAgentOptions(resume=...)``.
    """

    def __init__(self, model_id: str, *, cli_path: str | Path | None = None) -> None:
        self._model_id = model_id
        self._cli_path = cli_path
        self._session_base = get_daemon_state_dir() / "sdk-sessions"
        self._session_base.mkdir(parents=True, exist_ok=True)
        # Maps our session_id → SDK's ResultMessage.session_id (for resume=)
        self._sdk_session_ids: dict[str, str] = {}

    # jscpd:ignore-start (BenchmarkingAgent protocol implementation)
    async def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        tools: Sequence[ToolDefinition] = (),
        max_turns: int = 1,
        temperature: float | None = None,
        resume_session_id: str | None = None,
    ) -> AgentResult:
        # jscpd:ignore-end
        """Generate a response, optionally using tools over multiple turns."""
        if temperature is not None:
            logger.warning(
                "AnthropicBackend: temperature=%.2f ignored (SDK does not support it)",
                temperature,
            )

        if not tools and resume_session_id is None:
            return await self._generate_single_shot(system_prompt, user_prompt)

        return await self._generate_agentic(
            system_prompt, user_prompt, tools, max_turns, resume_session_id
        )

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
                cli_path=self._cli_path,
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
        resume_session_id: str | None,
    ) -> AgentResult:
        """Multi-turn agentic call via ``ClaudeSDKClient`` with MCP tools.

        Each session gets a persistent directory under ``sdk-sessions/`` so the
        SDK can be resumed natively. Concurrent clients are isolated by their
        unique ``XDG_DATA_HOME``.
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

        # Determine session directory and resume token
        if resume_session_id is not None:
            session_id = resume_session_id
            session_dir = self._session_base / session_id
            if not session_dir.exists():
                raise KeyError(f"Session directory for '{session_id}' not found")
            sdk_resume = self._sdk_session_ids.get(session_id)
        else:
            session_id = uuid.uuid4().hex
            session_dir = self._session_base / session_id
            session_dir.mkdir(parents=True)
            sdk_resume = None

        conv_result = _ConversationResult(
            answer="",
            usage=_UsageBreakdown(0, 0, 0, 0),
            reasoning_turns=0,
            sdk_session_id=None,
            history=(),
        )
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

            options = ClaudeAgentOptions(
                model=self._model_id,
                system_prompt=system_prompt,
                mcp_servers={"mem": memory_server},
                allowed_tools=allowed_tools,
                max_turns=max_turns + 1,  # +1 for final answer turn
                permission_mode="bypassPermissions",
                env={"XDG_DATA_HOME": str(session_dir)},
                resume=sdk_resume,
                cli_path=self._cli_path,
            )

            try:
                async with ClaudeSDKClient(options=options) as client:
                    conv_result = await self._run_sdk_conversation(client, user_prompt)
                break
            except Exception as exc:
                last_err = exc
                if attempt < _SDK_MAX_RETRIES:
                    logger.warning("SDK attempt %d failed: %s", attempt + 1, exc)
                    continue
                raise RuntimeError(
                    f"ClaudeSDKClient failed after {_SDK_MAX_RETRIES + 1} attempts"
                ) from last_err

        # Track the SDK session ID for future resume
        if conv_result.sdk_session_id is not None:
            self._sdk_session_ids[session_id] = conv_result.sdk_session_id

        answer = conv_result.answer
        if not answer.strip():
            answer = "I don't know."

        return make_agent_result(
            answer=answer,
            total_input=conv_result.usage.total_input,
            total_output=conv_result.usage.output_tokens,
            retrieved_tokens=retrieved_tokens,
            reasoning_turns=conv_result.reasoning_turns,
            elapsed=time.monotonic() - start_time,
            total_cost_usd=_compute_cost(self._model_id, conv_result.usage),
            history=conv_result.history,
            session_id=session_id,
        )

    @staticmethod
    async def _run_sdk_conversation(
        client: ClaudeSDKClient, user_prompt: str
    ) -> _ConversationResult:
        """Drive the SDK conversation and build MessageHistory from the stream."""
        await client.query(user_prompt)

        answer = ""
        usage = _UsageBreakdown(0, 0, 0, 0)
        reasoning_turns = 0
        sdk_session_id: str | None = None
        assistant_messages: list[AssistantMessage] = []

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                reasoning_turns += 1
                assistant_messages.append(message)
                for block in message.content:
                    if isinstance(block, TextBlock):
                        answer = block.text
                    elif isinstance(block, ToolUseBlock):
                        logger.debug(
                            "Tool call: %s(%s)",
                            block.name,
                            str(block.input)[:100],
                        )

            elif isinstance(message, ResultMessage):
                usage = _extract_usage(message)
                sdk_session_id = getattr(message, "session_id", None)

        history = _build_history_from_stream(user_prompt, assistant_messages)

        return _ConversationResult(
            answer=answer,
            usage=usage,
            reasoning_turns=reasoning_turns,
            sdk_session_id=sdk_session_id,
            history=history,
        )
