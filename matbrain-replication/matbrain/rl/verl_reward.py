"""verl custom reward entry point for Mat-T1.

    custom_reward_function.path=matbrain/rl/verl_reward.py
    custom_reward_function.name=compute_score

verl's reward manager decodes the full multi-turn response (assistant tokens
plus the interleaved tool-response tokens) and calls ``compute_score``. The
returned dict's ``score`` is the scalar reward; the other keys are logged as
reward_extra_info (turns / think / format / syntax curves as in Fig. 2g,h).
"""

from __future__ import annotations

from typing import Any

from matbrain.rl.rewards import compute_reward


def compute_score(data_source: str, solution_str: str, ground_truth: Any = None, extra_info: dict | None = None, **kwargs) -> dict[str, float]:
    prompt = ""
    if isinstance(extra_info, dict):
        prompt = str(extra_info.get("prompt_text", ""))
    breakdown = compute_reward(solution_str, prompt=prompt)
    out = breakdown.as_dict()
    # ``acc`` is what DAPO-style group filtering and verl's val metrics look at.
    out["acc"] = out["score"]
    return out
