"""Composite process reward for Mat-T1 (paper Methods, Eqs. 1-5).

    R_total = w1 R_turns + w2 R_think + w3 R_format + w4 R_syntax,
    w = (0.1, 0.3, 0.25, 0.35)

* R_turns  (Eq. 2): tiered reward on the number of tool-interaction turns n,
  0 / 0.5 / 0.7 / 0.85 / 1.0 at n = 0, ceil(k/4), ceil(k/2), ceil(3k/4), >= k; k = 4.
* R_think  (Eq. 3): tanh(L / lambda), L = mean token count inside <think>, lambda = 500.
* R_format (Eq. 4): alpha * mean_i 1[Think_i < Tool_i] + beta * 1[Think_final < Answer_final],
  alpha = 0.4, beta = 0.6.
* R_syntax (Eq. 5): fraction of tool calls that are JSON-parsable, registered
  in the Mat-MCP namespace and pass Pydantic parameter validation.

The reward is purely process-based: the SFT answers are *not* used as targets
during RL (the paper uses tool determinism as the validator instead).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable

from matbrain.agent.parsing import HANDLE_RE, Step, split_transcript

WEIGHTS = (0.1, 0.3, 0.25, 0.35)
TURNS_K = 4
THINK_LAMBDA = 500.0
FORMAT_ALPHA = 0.4
FORMAT_BETA = 0.6


# --------------------------------------------------------------------------- #
# Token counting for R_think
# --------------------------------------------------------------------------- #
_TOKENIZER: Any = None
_APPROX = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def set_tokenizer(tokenizer: Any) -> None:
    """Use a HF tokenizer (e.g. Qwen3's) for exact token counts in R_think."""
    global _TOKENIZER
    _TOKENIZER = tokenizer


@lru_cache(maxsize=1)
def _env_tokenizer():
    path = os.environ.get("MATBRAIN_REWARD_TOKENIZER")
    if not path:
        return None
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(path, trust_remote_code=True)


def count_tokens(text: str) -> int:
    tok = _TOKENIZER or _env_tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    # Fallback approximation (~1 token per word / punctuation mark), used only
    # when no tokenizer is configured, e.g. in unit tests.
    return len(_APPROX.findall(text))


# --------------------------------------------------------------------------- #
# Individual reward terms
# --------------------------------------------------------------------------- #
def turns_reward(n: int, k: int = TURNS_K) -> float:
    """Eq. 2. Values between the listed thresholds take the lower tier."""
    if n <= 0:
        return 0.0
    tiers = [(math.ceil(k / 4), 0.5), (math.ceil(k / 2), 0.7), (math.ceil(3 * k / 4), 0.85), (k, 1.0)]
    reward = 0.0
    for threshold, value in tiers:
        if n >= threshold:
            reward = max(reward, value)
    return reward


def think_reward(steps: list[Step], lam: float = THINK_LAMBDA, counter: Callable[[str], int] = count_tokens) -> tuple[float, float]:
    """Eq. 3. Returns (reward, L)."""
    blocks = [t for s in steps for t in s.thinks]
    if not blocks:
        return 0.0, 0.0
    L = sum(counter(t) for t in blocks) / len(blocks)
    return math.tanh(L / lam), L


def format_reward(steps: list[Step], alpha: float = FORMAT_ALPHA, beta: float = FORMAT_BETA) -> float:
    """Eq. 4. Intermediate steps = all assistant turns but the last."""
    if not steps:
        return 0.0
    intermediate, final = steps[:-1], steps[-1]
    inter = 0.0
    if intermediate:
        ok = [s.think_precedes_tool and s.answer is None for s in intermediate]
        inter = sum(ok) / len(ok)
    fin = 1.0 if final.think_precedes_answer else 0.0
    return alpha * inter + beta * fin


class _KnownHandles:
    """Artifact handles are only valid if produced earlier in the trajectory."""

    def __init__(self, handles: Iterable[str]):
        self.handles = set(handles)

    def exists(self, handle: str) -> bool:
        return handle.strip() in self.handles


@lru_cache(maxsize=1)
def _pool_registry():
    """The tool pool exposed during RL. MATBRAIN_TOOL_POOL (comma-separated names,
    printed by training/rl/make_tool_config.py) restricts it so that a call to a
    registered tool that was not offered to the policy does not count as valid."""
    from matbrain.mcp import load_registry

    registry = load_registry()
    pool = [t.strip() for t in os.environ.get("MATBRAIN_TOOL_POOL", "").split(",") if t.strip()]
    return registry.subset(include=pool) if pool else registry


def syntax_reward(steps: list[Step], registry=None, prompt: str = "", strict_handles: bool = True) -> tuple[float, int, int]:
    """Eq. 5. Returns (reward, n_valid, M)."""
    if registry is None:
        registry = _pool_registry()
    known = set(HANDLE_RE.findall(prompt))
    valid = total = 0
    for step in steps:
        for call in step.tool_calls:
            total += 1
            if call.json_ok:
                check = registry.validate(call.name, call.arguments, strict_handles=strict_handles, store=_KnownHandles(known))
                valid += int(check.ok)
        for obs in step.observations:
            known.update(HANDLE_RE.findall(obs))
    return (valid / total if total else 0.0), valid, total


# --------------------------------------------------------------------------- #
@dataclass
class RewardBreakdown:
    score: float
    turns: float
    think: float
    format: float
    syntax: float
    n_tool_turns: int
    n_tool_calls: int
    n_valid_calls: int
    mean_think_tokens: float
    n_steps: int

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


def compute_reward(
    transcript: str | list[Step],
    registry=None,
    prompt: str = "",
    weights: tuple[float, float, float, float] = WEIGHTS,
    k: int = TURNS_K,
    lam: float = THINK_LAMBDA,
    strict_handles: bool = True,
) -> RewardBreakdown:
    steps = split_transcript(transcript) if isinstance(transcript, str) else transcript
    n_tool_turns = sum(1 for s in steps if s.is_tool_step)
    r_turns = turns_reward(n_tool_turns, k)
    r_think, L = think_reward(steps, lam)
    r_format = format_reward(steps)
    r_syntax, n_valid, m = syntax_reward(steps, registry, prompt, strict_handles)
    w1, w2, w3, w4 = weights
    total = w1 * r_turns + w2 * r_think + w3 * r_format + w4 * r_syntax
    return RewardBreakdown(total, r_turns, r_think, r_format, r_syntax, n_tool_turns, m, n_valid, L, len(steps))
