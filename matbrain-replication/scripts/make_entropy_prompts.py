"""Turn benchmark tasks into prompts for the entropy analysis (Fig. 4).

    python scripts/make_entropy_prompts.py --tasks data/benchmark/tasks.jsonl --out data/entropy/r1_prompts.jsonl --n 200
    python scripts/make_entropy_prompts.py --tasks data/benchmark/tasks.jsonl --out data/entropy/t1_prompts.jsonl --n 200 --with-tools

``--with-tools`` adds the Mat-T1 system prompt and the tool schemas, so the
executive models are scored on tool-planning outputs; without it the prompts
use the Mat-R1 system prompt (direct analytical answers).
"""

from __future__ import annotations

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matbrain.data.formats import iter_records, write_jsonl  # noqa: E402
from matbrain.prompts import MAT_R1_SYSTEM, MAT_T1_SYSTEM  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--with-tools", action="store_true")
    ap.add_argument("--tools", nargs="*", help="tool pool (default: MATBRAIN_TOOL_POOL or all non-DB tools)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tasks = list(iter_records(args.tasks))
    tasks = random.Random(args.seed).sample(tasks, min(args.n, len(tasks)))
    tools = None
    if args.with_tools:
        from matbrain.eval.runner import benchmark_registry

        tools = benchmark_registry(args.tools).openai_tools(flat=True)
    system = MAT_T1_SYSTEM if args.with_tools else MAT_R1_SYSTEM
    rows = [{"id": t["id"], "messages": [{"role": "system", "content": system}, {"role": "user", "content": t["prompt"]}], **({"tools": tools} if tools else {})} for t in tasks]
    print(f"wrote {write_jsonl(args.out, rows)} prompts to {args.out}")


if __name__ == "__main__":
    main()
