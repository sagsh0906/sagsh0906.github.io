"""Offline end-to-end demo of the MatBrain loop (no GPU, no API keys).

Replays the paper's Fig. 5b case study - CIF of the mixed-halide perovskite
CsPb(Cl0.2Br0.4I0.4)3 - with *scripted* Mat-T1 / Mat-R1 policies, while every
tool call is executed for real by the Mat-MCP registry (pymatgen backends):

iteration 1  Mat-T1 expands the formula and calls CrystaLLM with the
             non-stoichiometric composition -> rejected by the tool.
Mat-R1       recognises a disordered mixed-halide problem -> CONTINUE: use the
             cubic end members + Vegard's law and a VCA occupancy model.
iteration 2  Mat-T1 interpolates the lattice parameter, builds the VCA
             structure, validates it and computes t / mu.
Mat-R1       FINISH -> the handle is resolved into the final CIF.

End-member lattice constants are the MP values quoted in Fig. 5b (the MP
search itself needs MP_API_KEY and network access).

    python scripts/demo_offline.py
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matbrain.agent.graph import MatBrain  # noqa: E402
from matbrain.agent.llm import ScriptedBackend  # noqa: E402
from matbrain.mcp import load_registry  # noqa: E402

END_MEMBERS = {"CsPbCl3": 5.67984, "CsPbBr3": 5.94863, "CsPbI3": 6.27514}
QUERY = "Provide the CIF file of CsPb(Cl0.2Br0.4I0.4)3 perovskite."


def call(name: str, **arguments) -> str:
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}) + "\n</tool_call>"


def mat_t1_policy(messages, tools) -> str:
    """Scripted executive policy keyed on the conversation state."""
    user = next(m["content"] for m in messages if m["role"] == "user")
    tool_msgs = [m["content"] for m in messages if m["role"] == "tool"]
    if "Vegard" not in user:  # iteration 1
        if not tool_msgs:
            return (
                "<think>The formula CsPb(Cl0.2Br0.4I0.4)3 expands to CsPbCl0.6Br1.2I1.2. Try to generate the structure "
                "directly with CrystaLLM.</think>\n" + call("crystallm_generate", formula="CsPbCl0.6Br1.2I1.2", num_samples=2)
            )
        return "<think>The generator rejected the fractional composition.</think>\n<answer>Direct generation failed: " + tool_msgs[-1] + "</answer>"
    # iteration 2 (after Mat-R1's feedback)
    handle = next((h for t in tool_msgs for h in re.findall(r"cif://[0-9a-f]{12}", t)), None)
    if len(tool_msgs) == 0:
        return (
            "<think>Apply Vegard's law to the cubic Pm-3m end-member lattice constants with the halide fractions "
            "0.2/0.4/0.4.</think>\n"
            + call("vegard_interpolate", end_member_values=END_MEMBERS, fractions={"CsPbCl3": 0.2, "CsPbBr3": 0.4, "CsPbI3": 0.4})
        )
    if len(tool_msgs) == 1:
        a = json.loads(tool_msgs[0])["result"]["value"]
        return (
            f"<think>a_VCA = {a} A. Build the Pm-3m VCA cell with X-site occupancies Cl/Br/I = 0.2/0.4/0.4 and "
            "compute the Goldschmidt and octahedral factors in parallel.</think>\n"
            + call("build_disordered_perovskite", a_site="Cs", b_site="Pb", x_site_occupancy={"Cl": 0.2, "Br": 0.4, "I": 0.4}, lattice_a=a)
            + "\n"
            + call("perovskite_factors", a_site={"Cs": 1}, b_site={"Pb": 1}, x_site={"Cl": 0.2, "Br": 0.4, "I": 0.4})
        )
    if len(tool_msgs) == 3:
        return "<think>Validate the accepted VCA structure.</think>\n" + call("validate_structure", cif=handle)
    return f"<think>The VCA structure is valid and charge balanced.</think>\n<answer>Accepted VCA structure: {handle}\n{tool_msgs[-1]}</answer>"


def mat_r1_policy(messages, tools) -> str:
    history = messages[-1]["content"]
    if "### Iteration 2" not in history:
        return (
            "<think>The tools reject non-stoichiometric formulas. This is a disordered mixed-halide ABX3 problem: Cl, "
            "Br and I statistically share the X sites. An average-crystal (VCA) description with fractional "
            "occupancies is appropriate, with the lattice parameter from Vegard's law.</think>\n"
            "<interpretation>Direct database retrieval / text-to-CIF generation is not applicable to the fractional "
            "composition.</interpretation>\n<decision>CONTINUE</decision>\n"
            "<next_instruction>Use the lattice parameters of the cubic Pm-3m end members CsPbCl3 (5.67984 A), CsPbBr3 "
            "(5.94863 A) and CsPbI3 (6.27514 A) to apply Vegard's law, then build and validate a VCA CIF with X-site "
            "occupancies Cl/Br/I = 0.2/0.4/0.4 and report tolerance and octahedral factors.</next_instruction>"
        )
    handle = re.findall(r"cif://[0-9a-f]{12}", history)[-1]
    return (
        "<think>a = 6.0255 A, occupancies normalised, CIF parseable, t and mu in the perovskite-forming range.</think>\n"
        "<interpretation>The VCA fractional-occupancy model is consistent with the target stoichiometry.</interpretation>\n"
        f"<decision>FINISH</decision>\n<answer>{handle}</answer>"
    )


async def main() -> None:
    brain = MatBrain(ScriptedBackend(mat_t1_policy, "Mat-T1"), ScriptedBackend(mat_r1_policy, "Mat-R1"), load_registry().subset(disable_target_db=True))
    result = await brain.arun(QUERY)
    for ex in result.executions:
        print(f"\n===== Mat-T1 iteration {ex['iteration']} =====\n{ex['summary']}")
    for i, an in enumerate(result.analyses, 1):
        print(f"\n===== Mat-R1 decision {i}: {an['decision']} =====\n{an['interpretation']}")
    print(f"\n===== Final answer (iterations={result.iterations}, forced={result.forced}) =====\n{result.answer}")


if __name__ == "__main__":
    asyncio.run(main())
