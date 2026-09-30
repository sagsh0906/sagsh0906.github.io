"""Execution node (Mat-T1): Think-Plan-Execute-Observe loop over Mat-MCP tools.

Includes the *Runtime Parameter Validation Layer*: every tool call proposed by
the model is checked against the registry and its Pydantic schema (including
CIF parseability / artifact-handle existence) and only validated calls are
dispatched. Results and errors are turned into standardised observations.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from typing import Any

from matbrain.agent.llm import ChatBackend
from matbrain.agent.parsing import HANDLE_RE, TOOL_CALL_RE, parse_step
from matbrain.mcp.registry import ToolRegistry, ToolResult
from matbrain.prompts import MAT_T1_SYSTEM

BUDGET_EXHAUSTED = (
    "The tool-call budget is exhausted. Do not call more tools. Summarise the evidence gathered so far and "
    "give the final result inside <answer></answer>."
)


@dataclass
class CallRecord:
    name: str | None
    arguments: Any
    valid: bool
    ok: bool
    observation: str
    error: str | None = None
    elapsed: float = 0.0


@dataclass
class ExecutionTrace:
    instruction: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    turns: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    stop_reason: str = ""
    n_tool_calls: int = 0
    n_rejected: int = 0
    n_failed: int = 0

    def summary(self, max_obs_chars: int = 1500) -> str:
        """Compact execution history handed to the reasoning node."""
        lines = [f"Instruction: {self.instruction}"]
        for i, t in enumerate(self.turns, 1):
            for c in t["calls"]:
                args = json.dumps(c["arguments"], ensure_ascii=False, default=str)
                if len(args) > 400:
                    args = args[:400] + "...}"
                obs = c["observation"]
                if len(obs) > max_obs_chars:
                    obs = obs[: max_obs_chars // 2] + " ...[truncated]... " + obs[-max_obs_chars // 2 :]
                lines.append(f"[turn {i}] call {c['name']}({args})\n  -> {obs}")
        lines.append(f"Mat-T1 result ({self.stop_reason}): {self.answer or '(no answer)'}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RemoteMCPTools:
    """Dispatch validated calls to a running Mat-MCP server (streamable HTTP)."""

    def __init__(self, url: str, registry: ToolRegistry):
        self.url = url
        self.registry = registry
        self._stack: AsyncExitStack | None = None
        self._session = None

    async def __aenter__(self):
        from mcp import ClientSession

        try:
            from mcp.client.streamable_http import streamable_http_client as _client
        except ImportError:  # mcp 1.x
            from mcp.client.streamable_http import streamablehttp_client as _client
        self._stack = AsyncExitStack()
        streams = await self._stack.enter_async_context(_client(self.url))
        self._session = await self._stack.enter_async_context(ClientSession(streams[0], streams[1]))
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    async def acall(self, name: str, arguments: dict, **_: Any) -> ToolResult:
        res = await self._session.call_tool(name, arguments)
        text = " ".join(getattr(p, "text", "") for p in res.content)
        try:
            payload = json.loads(text)
            ok = payload.get("status") == "success"
            return ToolResult(name=name, ok=ok, data=payload.get("result"), error=payload.get("error"))
        except (json.JSONDecodeError, AttributeError):
            return ToolResult(name=name, ok=not getattr(res, "is_error", False), data=text)


class _SeenHandles:
    """Handle registry for remote dispatch: artifacts live in the server's store,
    so a handle is accepted if it appeared in the prompt or an earlier observation."""

    def __init__(self, text: str = ""):
        self.handles = set(HANDLE_RE.findall(text))

    def update(self, text: str) -> None:
        self.handles.update(HANDLE_RE.findall(text))

    def exists(self, handle: str) -> bool:
        return handle.strip() in self.handles


class ToolExecutor:
    def __init__(
        self,
        backend: ChatBackend,
        registry: ToolRegistry,
        max_tool_calls: int = 12,
        max_turns: int = 8,
        max_observation_chars: int = 4000,
        system_prompt: str = MAT_T1_SYSTEM,
        dispatcher: Any = None,
        temperature: float | None = 0.0,
    ):
        self.backend = backend
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.max_turns = max_turns
        self.max_observation_chars = max_observation_chars
        self.system_prompt = system_prompt
        self.dispatcher = dispatcher or registry
        self.temperature = temperature
        self.tools = registry.openai_tools()

    async def _chat(self, messages, tools):
        kwargs = {} if self.temperature is None else {"temperature": self.temperature}
        return await self.backend.chat(messages, tools=tools, **kwargs)

    async def _dispatch(self, calls, seen: _SeenHandles | None = None) -> list[CallRecord]:
        records: list[CallRecord | None] = [None] * len(calls)
        pending = []
        for i, call in enumerate(calls):
            if not call.json_ok:
                records[i] = CallRecord(call.name, call.raw, False, False, json.dumps({"status": "error", "error": call.parse_error}), call.parse_error)
                continue
            store = None if self.dispatcher is self.registry else seen
            check = self.registry.validate(call.name, call.arguments, strict_handles=True, store=store)
            if not check.ok:
                records[i] = CallRecord(call.name, call.arguments, False, False, json.dumps({"tool": call.name, "status": "error", "error": check.error}), check.error)
                continue
            pending.append((i, call))
        results = await asyncio.gather(*(self.dispatcher.acall(c.name, c.arguments, strict_handles=True) for _, c in pending))
        for (i, call), res in zip(pending, results):
            records[i] = CallRecord(call.name, call.arguments, True, res.ok, res.to_observation(self.max_observation_chars), res.error, res.elapsed)
        return [r for r in records if r is not None]

    async def run(self, instruction: str, context: str | None = None) -> ExecutionTrace:
        user = instruction if not context else f"{instruction}\n\nContext from previous iterations:\n{context}"
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": user}]
        trace = ExecutionTrace(instruction=instruction, messages=messages)
        seen = _SeenHandles(user)
        for _ in range(self.max_turns):
            budget_left = self.max_tool_calls - trace.n_tool_calls
            reply = await self._chat(messages, self.tools if budget_left > 0 else None)
            text = reply.full_text()
            step = parse_step(text)
            calls = step.tool_calls
            if not calls or budget_left <= 0:
                messages.append({"role": "assistant", "content": text})
                stripped = TOOL_CALL_RE.sub("", re.sub(r"<think>.*?</think>", "", text, flags=re.S)).strip()
                trace.answer = step.answer if step.answer is not None else stripped
                trace.stop_reason = "answer" if step.answer is not None else ("budget" if calls else "no_action")
                break
            calls = calls[:budget_left]
            records = await self._dispatch(calls, seen)
            for r in records:
                seen.update(r.observation)
            trace.n_tool_calls += len(calls)
            trace.n_rejected += sum(not r.valid for r in records)
            trace.n_failed += sum(r.valid and not r.ok for r in records)
            content = TOOL_CALL_RE.sub("", text).strip()
            tool_calls = [
                {"id": f"call_{len(trace.turns)}_{j}", "type": "function", "function": {"name": r.name or "invalid", "arguments": r.arguments if isinstance(r.arguments, str) else json.dumps(r.arguments, ensure_ascii=False)}}
                for j, r in enumerate(records)
            ]
            messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
            for tc, r in zip(tool_calls, records):
                messages.append({"role": "tool", "tool_call_id": tc["id"], "content": r.observation})
            trace.turns.append({"assistant": text, "calls": [asdict(r) for r in records]})
            if trace.n_tool_calls >= self.max_tool_calls:
                messages.append({"role": "user", "content": BUDGET_EXHAUSTED})
        else:
            trace.stop_reason = "max_turns"
        return trace
