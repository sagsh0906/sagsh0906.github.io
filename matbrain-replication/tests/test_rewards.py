import json
import math

import pytest

from matbrain.agent.parsing import split_transcript
from matbrain.rl.rewards import WEIGHTS, compute_reward, format_reward, think_reward, turns_reward
from matbrain.rl.verl_reward import compute_score


def tc(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def resp(*bodies):
    return "".join(f"<tool_response>\n{b}\n</tool_response>\n" for b in bodies)


def test_turns_reward_tiers():
    assert [turns_reward(n) for n in range(6)] == [0.0, 0.5, 0.7, 0.85, 1.0, 1.0]
    assert turns_reward(3, k=8) == 0.5 and turns_reward(4, k=8) == 0.7 and turns_reward(6, k=8) == 0.85


def test_think_reward_tanh():
    steps = split_transcript("<think>" + " w" * 500 + "</think><answer>x</answer>")
    r, L = think_reward(steps)
    assert L == 500 and r == pytest.approx(math.tanh(1.0)) and r == pytest.approx(0.7616, abs=1e-4)


def test_format_reward_cases():
    good = "<think>a</think>" + tc("check_charge_balance", formula="NaCl") + "user\n" + resp("{}") + "assistant\n<think>b</think><answer>ok</answer>"
    assert format_reward(split_transcript(good)) == pytest.approx(1.0)
    tool_before_think = tc("check_charge_balance", formula="NaCl") + "<think>a</think>" + resp("{}") + "<think>b</think><answer>ok</answer>"
    assert format_reward(split_transcript(tool_before_think)) == pytest.approx(0.6)
    premature_answer = "<think>a</think><answer>early</answer>" + tc("check_charge_balance", formula="NaCl") + resp("{}") + "<think>b</think><answer>ok</answer>"
    assert format_reward(split_transcript(premature_answer)) == pytest.approx(0.6)
    no_final_think = "<think>a</think>" + tc("check_charge_balance", formula="NaCl") + resp("{}") + "<answer>ok</answer>"
    assert format_reward(split_transcript(no_final_think)) == pytest.approx(0.4)


def test_composite_reward_and_handle_provenance():
    h = "cif://08f313fc8961"
    t = (
        "<think>" + " p" * 500 + "</think>" + tc("build_disordered_perovskite", a_site="Cs", b_site="Pb", x_site_occupancy={"Cl": 0.2, "Br": 0.4, "I": 0.4}, lattice_a=6.0255)
        + "<|im_end|>\n<|im_start|>user\n" + resp(json.dumps({"result": {"handle": h}})) + "<|im_end|>\n<|im_start|>assistant\n"
        + "<think>" + " q" * 500 + "</think>" + tc("validate_structure", cif=h) + tc("validate_structure", cif="cif://aaaaaaaaaaaa")
        + "\n" + resp("{}", "{}")
        + "<think>" + " r" * 500 + "</think><answer>done</answer>"
    )
    b = compute_reward(t)
    assert b.n_tool_turns == 2 and b.n_tool_calls == 3 and b.n_valid_calls == 2
    expected = WEIGHTS[0] * 0.7 + WEIGHTS[1] * math.tanh(1) + WEIGHTS[2] * 1.0 + WEIGHTS[3] * (2 / 3)
    assert b.score == pytest.approx(expected)


def test_syntax_reward_rejects_bad_calls():
    t = (
        "<think>x</think><tool_call>{not json}</tool_call>" + resp("e")
        + "<think>x</think>" + tc("no_such_tool", a=1) + resp("e")
        + "<think>x</think>" + tc("check_charge_balance", formula="NaCl", extra=1) + resp("e")
        + "<think>x</think>" + tc("check_charge_balance", formula="NaCl") + resp("ok")
        + "<think>x</think><answer>a</answer>"
    )
    b = compute_reward(t)
    assert b.n_tool_calls == 4 and b.n_valid_calls == 1 and b.syntax == pytest.approx(0.25)
    assert b.turns == 1.0


def test_no_tool_use_gets_no_turn_or_syntax_credit():
    b = compute_reward("<think>" + " w" * 50 + "</think><answer>guess</answer>")
    assert b.turns == 0 and b.syntax == 0 and b.format == pytest.approx(0.6)


def test_verl_entry_point():
    out = compute_score("mat20k_rl", "<think>a</think><answer>b</answer>", "", {"prompt_text": ""})
    assert set(out) >= {"score", "acc", "turns", "think", "format", "syntax"}
    assert out["score"] == out["acc"]
