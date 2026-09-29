import asyncio
import json
import os
import sys

import pytest

from matbrain.agent.executor import ToolExecutor
from matbrain.agent.graph import MatBrain, parse_decision
from matbrain.agent.llm import ScriptedBackend

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


def call(name, **args):
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": args}) + "\n</tool_call>"


def test_parse_decision():
    d = parse_decision("<think>x</think><interpretation>ok</interpretation><decision>CONTINUE</decision><next_instruction>do y</next_instruction>")
    assert d.decision == "CONTINUE" and d.next_instruction == "do y"
    d = parse_decision("<think>x</think>the answer is 42")
    assert d.decision == "FINISH" and d.answer == "the answer is 42"


def test_executor_validation_layer_and_budget(registry):
    replies = [
        "<think>a</think>" + call("no_such_tool") + call("check_charge_balance", formula="NaCl"),
        "<think>b</think>" + call("check_charge_balance", formula="MgO") + call("check_charge_balance", formula="CaO"),
        "<think>c</think>" + call("check_charge_balance", formula="KCl"),
        "<think>d</think><answer>done</answer>",
    ]
    ex = ToolExecutor(ScriptedBackend(replies), registry, max_tool_calls=3)
    trace = asyncio.run(ex.run("check some salts"))
    assert trace.n_tool_calls == 3 and trace.n_rejected == 1
    assert "not registered" in trace.turns[0]["calls"][0]["observation"]
    assert trace.turns[0]["calls"][1]["ok"]
    assert trace.stop_reason == "budget"  # third turn's call exceeds the 3-call budget


def test_matbrain_fig5b_replay(registry):
    import demo_offline as demo

    brain = MatBrain(ScriptedBackend(demo.mat_t1_policy), ScriptedBackend(demo.mat_r1_policy), registry.subset(disable_target_db=True))
    res = asyncio.run(brain.arun(demo.QUERY))
    assert res.iterations == 2 and not res.forced
    assert "_cell_length_a   6.025476" in res.answer
    first = res.executions[0]["trace"]["turns"][0]["calls"][0]
    assert not first["ok"] and "Non-stoichiometric" in first["observation"]


def test_max_iterations_forces_answer(registry):
    t1 = ScriptedBackend(lambda m, t: "<think>x</think><answer>nothing new</answer>")
    r1_replies = lambda m, t: (
        "<decision>FINISH</decision><answer>best effort</answer>" if "maximum number of iterations" in m[-1]["content"]
        else "<decision>CONTINUE</decision><next_instruction>try again</next_instruction>"
    )
    brain = MatBrain(t1, ScriptedBackend(r1_replies), registry, max_iterations=3)
    res = asyncio.run(brain.arun("q"))
    assert res.iterations == 3 and res.forced and res.answer == "best effort"


def test_initial_analysis_can_finish_without_tools(registry):
    t1 = ScriptedBackend([])  # must never be called
    r1 = ScriptedBackend(["<decision>FINISH</decision><answer>Fm-3m</answer>"])
    brain = MatBrain(t1, r1, registry, start_with_analysis=True)
    res = asyncio.run(brain.arun("What is the space group of rocksalt NaCl?"))
    assert res.answer == "Fm-3m" and res.iterations == 0


@pytest.mark.parametrize("mode", ["decision_tree"])
def test_runner_decision_tree_offline(mode, registry, nacl):
    from matbrain.chem import structure_to_cif
    from matbrain.data.benchmark import build_tasks
    from matbrain.eval.runner import run_one

    r = {"material_id": "mp-1", "cif": structure_to_cif(nacl), "is_metal": False, "is_magnetic": False, "formation_energy_per_atom": -2.1, "spacegroup_symbol": "Fm-3m", "spacegroup_number": 225, "crystal_system": "cubic"}
    tasks = build_tasks([r], properties=["is_magnetic", "formation_energy_per_atom"])
    pool = registry.subset(disable_target_db=True)
    outs = [asyncio.run(run_one(t, mode, {}, pool)) for t in tasks]
    assert all("<answer>" in o["output"] for o in outs)
