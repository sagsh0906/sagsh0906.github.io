"""Parsing of Think-then-Act transcripts.

Mat-T1 emits, per assistant turn, ``<think>...</think>`` followed by either
one or more hermes-style ``<tool_call>{"name": ..., "arguments": {...}}</tool_call>``
blocks (intermediate step) or an ``<answer>...</answer>`` block (final step).
Tool observations come back as ``<tool_response>...</tool_response>`` inside
a user turn (Qwen3 chat template).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)
TOOL_RESPONSE_RE = re.compile(r"<tool_response>(.*?)</tool_response>", re.S)
# One user turn may carry several tool responses (parallel calls).
TOOL_RESPONSE_BLOCK_RE = re.compile(r"(?:<tool_response>.*?</tool_response>\s*)+", re.S)
SPECIAL_TOKENS_RE = re.compile(r"<\|im_(?:start|end)\|>|<\|endoftext\|>")
HANDLE_RE = re.compile(r"cif://[0-9a-f]{12}")


@dataclass
class ToolCall:
    name: str | None
    arguments: Any
    raw: str
    parse_error: str | None = None

    @property
    def json_ok(self) -> bool:
        return self.parse_error is None


@dataclass
class Step:
    """One assistant turn."""

    text: str
    thinks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    answer: str | None = None
    think_end: int | None = None  # char offset of the first </think>
    first_tool_start: int | None = None
    answer_start: int | None = None
    # tool responses that followed this step (observations)
    observations: list[str] = field(default_factory=list)

    @property
    def has_think(self) -> bool:
        return bool(self.thinks)

    @property
    def is_tool_step(self) -> bool:
        return bool(self.tool_calls)

    @property
    def think_precedes_tool(self) -> bool:
        return self.has_think and self.is_tool_step and self.think_end is not None and self.think_end <= self.first_tool_start

    @property
    def think_precedes_answer(self) -> bool:
        return self.has_think and self.answer is not None and self.think_end is not None and self.think_end <= self.answer_start


def parse_tool_call(raw: str) -> ToolCall:
    body = raw.strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        return ToolCall(None, None, raw, f"tool call is not valid JSON: {exc}")
    if not isinstance(obj, dict) or "name" not in obj:
        return ToolCall(None, None, raw, "tool call must be a JSON object with a 'name' field")
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError as exc:
            return ToolCall(obj["name"], args, raw, f"arguments string is not valid JSON: {exc}")
    return ToolCall(str(obj["name"]), args, raw)


def parse_step(text: str) -> Step:
    step = Step(text=text)
    thinks = list(THINK_RE.finditer(text))
    # Qwen3 in thinking mode may start generation *inside* <think> (the opening
    # tag is part of the prompt); treat a leading "...</think>" as a think block.
    if not thinks and "</think>" in text:
        end = text.index("</think>")
        step.thinks = [text[:end]]
        step.think_end = end
    elif thinks:
        step.thinks = [m.group(1) for m in thinks]
        step.think_end = thinks[0].end()
    calls = list(TOOL_CALL_RE.finditer(text))
    if calls:
        step.tool_calls = [parse_tool_call(m.group(1)) for m in calls]
        step.first_tool_start = calls[0].start()
    ans = ANSWER_RE.search(text)
    if ans:
        step.answer = ans.group(1).strip()
        step.answer_start = ans.start()
    return step


def _clean_segment(seg: str) -> str:
    seg = SPECIAL_TOKENS_RE.sub("", seg)
    # role headers left behind when special tokens are skipped during decoding
    seg = re.sub(r"^\s*assistant\s*\n", "", seg)
    seg = re.sub(r"\n\s*user\s*$", "", seg.rstrip())
    return seg.strip()


def split_transcript(transcript: str) -> list[Step]:
    """Split a decoded multi-turn response (as seen by the verl reward manager)
    into assistant steps, attaching the tool observations that followed each."""
    steps: list[Step] = []
    pos = 0
    for block in TOOL_RESPONSE_BLOCK_RE.finditer(transcript):
        seg = _clean_segment(transcript[pos : block.start()])
        step = parse_step(seg)
        step.observations = [m.group(1).strip() for m in TOOL_RESPONSE_RE.finditer(block.group(0))]
        steps.append(step)
        pos = block.end()
    tail = _clean_segment(transcript[pos:])
    if tail or not steps:
        steps.append(parse_step(tail))
    return steps


def steps_from_messages(messages: list[dict[str, Any]]) -> list[Step]:
    """Build steps from an OpenAI-style message list (MatBrain executor traces)."""
    steps: list[Step] = []
    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            text = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or msg.get("reasoning")
            if reasoning and "<think>" not in text:
                text = f"<think>{reasoning}</think>\n{text}"
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                text += "\n<tool_call>\n" + json.dumps({"name": fn.get("name"), "arguments": json.loads(args) if isinstance(args, str) and args.strip().startswith("{") else args}) + "\n</tool_call>"
            steps.append(parse_step(text))
        elif role == "tool" and steps:
            steps[-1].observations.append(str(msg.get("content", "")))
    return steps


def extract_answer(text: str) -> str | None:
    m = ANSWER_RE.findall(text)
    return m[-1].strip() if m else None
