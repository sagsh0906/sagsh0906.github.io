"""Build Mat-20K-RL (verl parquet) from structure-generation / property-prediction
queries of Mat-252K-SFT.

    python -m matbrain.data.rl_dataset --sft data/sft_raw/mp.jsonl --out-dir data/rl --n 20000

Only the *queries* are used: SFT answers are not supervision targets during RL.
Queries are rewritten as open-ended tool tasks, input CIFs are registered as
artifact handles (persisted under MATBRAIN_ARTIFACT_DIR so the tool workers
can resolve them) and "high-complexity" queries are preferred: tasks that need
generation + validation + relaxation / property evaluation rather than a
single lookup (the paper's "no shortcut learning" design).
"""

from __future__ import annotations

import argparse
import os
import random
import re
from typing import Any

from matbrain.data.formats import iter_records, to_messages
from matbrain.prompts import MAT_T1_SYSTEM

PROPERTY_TASK = (
    "Determine the {prop} of the material whose structure is stored as {handle}. Do not guess: obtain the value "
    "with the Mat-MCP tools (validate the structure, relax it with a universal potential, then evaluate the "
    "property), and report the value with the method used.\n\nCIF for reference:\n{cif}"
)
GENERATION_TASK = (
    "{question}\nUse the Mat-MCP tools to generate candidates, validate them (parseability, interatomic "
    "distances, charge balance, symmetry) and relax the best one. Report the final structure handle and why it "
    "satisfies the target."
)
PROP_NAMES = {
    "formation_energy_per_atom": "formation energy per atom",
    "energy_above_hull": "energy above hull",
    "band_gap": "band gap",
    "efermi": "Fermi energy",
    "is_metal": "metallicity",
    "is_magnetic": "magnetic state",
    "ordering": "magnetic ordering",
}


def complexity(task: str) -> int:
    """Heuristic ranking: generation > multi-step properties > single-model properties."""
    if task == "structure_generation":
        return 3
    if task in ("property:energy_above_hull", "property:ordering", "property:is_magnetic"):
        return 2
    return 1


def to_rl_row(rec: dict[str, Any], index: int, split: str, store) -> dict[str, Any] | None:
    msgs = to_messages(rec)
    if not msgs:
        return None
    question = next(m["content"] for m in msgs if m["role"] == "user")
    task = rec.get("task", "")
    if task.startswith("property:"):
        m = re.search(r"^(data_.*)", question, re.S | re.M)  # CIF is appended after the question
        if not m:
            return None
        cif = m.group(1).strip()
        try:
            handle = store.put_cif(cif, {"source": "mat20k_rl", "material_id": rec.get("material_id")})
        except Exception:
            return None
        prompt_text = PROPERTY_TASK.format(prop=PROP_NAMES.get(task.split(":", 1)[1], task), handle=handle, cif=cif)
    elif task == "structure_generation":
        prompt_text = GENERATION_TASK.format(question=question.replace("Output a complete CIF.", "").strip())
    else:
        return None
    return {
        "data_source": "mat20k_rl",
        "agent_name": "tool_agent",
        "prompt": [{"role": "system", "content": MAT_T1_SYSTEM}, {"role": "user", "content": prompt_text}],
        "ability": "materials_tool_use",
        # Not used by the process reward; kept for analysis only.
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {"index": index, "split": split, "task": task, "material_id": rec.get("material_id", ""), "prompt_text": prompt_text},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", required=True, help="MP-grounded SFT jsonl (generate_distil mp output)")
    ap.add_argument("--out-dir", default="data/rl")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--val", type=int, default=200)
    ap.add_argument("--artifact-dir", default=os.environ.get("MATBRAIN_ARTIFACT_DIR", "data/rl/artifacts"))
    ap.add_argument("--tasks", nargs="*", help="keep only these task prefixes, e.g. property: (skip structure_generation when CrystaLLM is not installed)")
    ap.add_argument("--uniform", action="store_true", help="sample tasks uniformly instead of preferring high-complexity ones")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import pandas as pd

    from matbrain.mcp.artifacts import ArtifactStore

    store = ArtifactStore(args.artifact_dir)
    rng = random.Random(args.seed)
    recs = [r for r in iter_records(args.sft) if r.get("task") and (not args.tasks or r["task"].startswith(tuple(args.tasks)))]
    rng.shuffle(recs)
    if not args.uniform:
        recs.sort(key=lambda r: -complexity(r["task"]))  # stable: shuffled within tiers
    rows = []
    for rec in recs:
        row = to_rl_row(rec, len(rows), "train", store)
        if row:
            rows.append(row)
        if len(rows) >= args.n + args.val:
            break
    rng.shuffle(rows)
    val, train = rows[: args.val], rows[args.val :]
    for r in val:
        r["extra_info"]["split"] = "val"
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(train).to_parquet(os.path.join(args.out_dir, "mat20k_rl_train.parquet"))
    pd.DataFrame(val).to_parquet(os.path.join(args.out_dir, "mat20k_rl_val.parquet"))
    print(f"train={len(train)} val={len(val)}; artifacts in {args.artifact_dir} (set MATBRAIN_ARTIFACT_DIR for the tool workers)")


if __name__ == "__main__":
    main()
