"""Chat backends.

* ``OpenAIChatBackend``: any OpenAI-compatible endpoint (vLLM serving Mat-R1 /
  Mat-T1, DeepSeek / GPT / Gemini baselines, data-generation teachers).
* ``ScriptedBackend``: deterministic replies for tests and offline demos.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol


@dataclass
class ChatReply:
    content: str
    reasoning: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)  # [{"name":..., "arguments": {...}|str}]
    usage: dict[str, int] = field(default_factory=dict)
    logprobs: Any = None

    def full_text(self) -> str:
        """Assistant turn in the Think-then-Act text format used for training/rewards."""
        text = self.content or ""
        if self.reasoning and "<think>" not in text:
            text = f"<think>{self.reasoning}</think>\n{text}"
        for tc in self.tool_calls:
            if f'"{tc["name"]}"' in text and "<tool_call>" in text:
                continue
            args = tc["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            text += "\n<tool_call>\n" + json.dumps({"name": tc["name"], "arguments": args}, ensure_ascii=False) + "\n</tool_call>"
        return text


class ChatBackend(Protocol):
    name: str

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict] | None = None, **kwargs) -> ChatReply: ...


class OpenAIChatBackend:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        name: str | None = None,
    ):
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), timeout=timeout, max_retries=max_retries)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_body = extra_body or {}
        self.name = name or model

    async def chat(self, messages, tools=None, **kwargs) -> ChatReply:
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.pop("temperature", self.temperature),
        }
        max_tokens = kwargs.pop("max_tokens", self.max_tokens)
        if max_tokens:
            params["max_tokens"] = max_tokens
        if tools:
            params["tools"] = tools
        extra = {**self.extra_body, **kwargs.pop("extra_body", {})}
        if extra:
            params["extra_body"] = extra
        params.update(kwargs)
        resp = await self.client.chat.completions.create(**params)
        choice = resp.choices[0]
        msg = choice.message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        calls = [{"name": tc.function.name, "arguments": tc.function.arguments} for tc in (msg.tool_calls or [])]
        usage = resp.usage.model_dump() if getattr(resp, "usage", None) else {}
        return ChatReply(content=msg.content or "", reasoning=reasoning, tool_calls=calls, usage=usage, logprobs=getattr(choice, "logprobs", None))


class ScriptedBackend:
    """Replies from a list (consumed in order) or a callable(messages, tools) -> str."""

    def __init__(self, replies: list[str] | Callable[[list[dict], list[dict] | None], str], name: str = "scripted"):
        self.replies = replies
        self.name = name
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=None, **kwargs) -> ChatReply:
        self.calls.append(list(messages))
        await asyncio.sleep(0)
        if callable(self.replies):
            text = self.replies(messages, tools)
        else:
            if not self.replies:
                raise RuntimeError(f"{self.name}: no scripted replies left")
            text = self.replies.pop(0)
        return ChatReply(content=text)


def backend_from_config(cfg: dict[str, Any]) -> OpenAIChatBackend:
    """Build a backend from a YAML/dict config; ``api_key_env`` names the env variable."""
    cfg = dict(cfg)
    key_env = cfg.pop("api_key_env", None)
    if key_env:
        cfg["api_key"] = os.environ.get(key_env)
    return OpenAIChatBackend(**cfg)
