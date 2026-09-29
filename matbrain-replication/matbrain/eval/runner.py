"""Benchmark runner for the three evaluation settings of Fig. 3.

  direct       tool-free answering (Mat-R1, its base model, general LLMs)      Fig. 3a-c
  tool         shared Mat-MCP tool pool, 12-call budget, temperature 0         Fig. 3d-f
  matbrain     full Mat-R1 + Mat-T1 collaboration                             Fig. 3g-i
  decision_tree  scripted controller over the same tool pool                  Fig. 3d-f

Target-database retrieval (MP / OQMD) and web/literature search are removed
from the tool pool; the input CIF is registered as an artifact handle so every
tool-using system receives identical inputs.

    python -m matbrain.eval.runner --tasks data/benchmark/tasks.jsonl --config configs/benchmark.yaml \
        --system Mat-T1 --mode tool --out-dir results/benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import warnings
from typing import Any

import yaml

from matbrain.agent.executor import ToolExecutor
from matbrain.agent.graph import MatBrain
from matbrain.agent.llm import ChatBackend, backend_from_config
from matbrain.data.formats import iter_records
from matbrain.eval.decision_tree import DecisionTreeController
from matbrain.eval.metrics import score_predictions
from matbrain.mcp import load_registry
from matbrain.mcp.artifacts import get_store
from matbrain.mcp.registry import ANALYSIS, COMPUTATION, GENERATION, VALIDATION, ToolRegistry
from matbrain.prompts import MAT_R1_SYSTEM

BENCHMARK_CATEGORIES = [GENERATION, VALIDATION, ANALYSIS, COMPUTATION]


def benchmark_registry() -> ToolRegistry:
    if not os.environ.get("MATBRAIN_REFERENCE_ENTRIES"):
        warnings.warn(
            "MATBRAIN_REFERENCE_ENTRIES is not set: phase_diagram_ehull would query live MP reference entries, "
            "which can contain held-out materials. Build a training-split snapshot with scripts/build_reference_entries.py."
        )
    return load_registry().subset(categories=BENCHMARK_CATEGORIES, disable_target_db=True)


def tool_prompt(task: dict[str, Any]) -> str:
    if task.get("input_cif"):
        handle = get_store().put_cif(task["input_cif"], {"source": "benchmark_input"})
        return f"{task['prompt']}\n\nThe input structure is also available to the tools as artifact handle {handle}."
    return task["prompt"]


async def run_one(task: dict[str, Any], mode: str, system: dict[str, Any], registry: ToolRegistry) -> dict[str, Any]:
    if mode == "direct":
        backend: ChatBackend = system["backend"]
        messages = ([{"role": "system", "content": system["system_prompt"]}] if system.get("system_prompt") else []) + [{"role": "user", "content": task["prompt"]}]
        reply = await backend.chat(messages, temperature=0.0)
        return {"output": reply.content, "reasoning": reply.reasoning}
    if mode == "tool":
        ex = ToolExecutor(system["backend"], registry, max_tool_calls=12)
        trace = await ex.run(tool_prompt(task))
        return {"output": f"<answer>{trace.answer or ''}</answer>", "n_tool_calls": trace.n_tool_calls, "n_rejected": trace.n_rejected, "n_failed": trace.n_failed, "stop_reason": trace.stop_reason}
    if mode == "matbrain":
        brain = MatBrain(system["executor"], system["reasoner"], registry, max_iterations=system.get("max_iterations", 6), max_tool_calls=12)
        res = await brain.arun(tool_prompt(task))
        n_calls = sum(e["trace"]["n_tool_calls"] for e in res.executions)
        return {"output": f"<answer>{res.answer}</answer>", "iterations": res.iterations, "forced": res.forced, "n_tool_calls": n_calls}
    if mode == "decision_tree":
        ctl = DecisionTreeController(registry, max_tool_calls=12)
        out = await ctl.run(task)
        return {"output": out, "n_tool_calls": ctl.calls, "tool_log": ctl.log}
    raise ValueError(mode)


def load_system(cfg: dict[str, Any], name: str, mode: str) -> dict[str, Any]:
    if mode == "decision_tree":
        return {}
    spec = cfg["systems"][name]
    if mode == "matbrain":
        return {"executor": backend_from_config(cfg["systems"][spec["executor"]]["backend"]), "reasoner": backend_from_config(cfg["systems"][spec["reasoner"]]["backend"]), "max_iterations": spec.get("max_iterations", 6)}
    system_prompt = MAT_R1_SYSTEM if spec.get("system_prompt") == "mat_r1" else spec.get("system_prompt")
    return {"backend": backend_from_config(spec["backend"]), "system_prompt": system_prompt}


async def run_all(tasks, mode, system, registry, concurrency: int, out_path: str) -> dict[str, str]:
    done = {r["id"]: r for r in iter_records(out_path)} if os.path.exists(out_path) else {}
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()

    async def one(t):
        async with sem:
            try:
                res = await run_one(t, mode, system, registry)
            except Exception as exc:
                res = {"output": "", "error": f"{type(exc).__name__}: {exc}"}
        async with lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"id": t["id"], **res}, ensure_ascii=False, default=str) + "\n")
        return t["id"], res

    todo = [t for t in tasks if t["id"] not in done]
    results = await asyncio.gather(*(one(t) for t in todo))
    outputs = {k: v.get("output", "") for k, v in done.items()}
    outputs.update({k: v.get("output", "") for k, v in results})
    return outputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--config", default="configs/benchmark.yaml")
    ap.add_argument("--system", default="decision-tree")
    ap.add_argument("--mode", choices=["direct", "tool", "matbrain", "decision_tree"], required=True)
    ap.add_argument("--families", nargs="*", help="subset: classification regression structure_design")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--out-dir", default="results/benchmark")
    args = ap.parse_args()

    tasks = [t for t in iter_records(args.tasks) if not args.families or t["family"] in args.families]
    tasks = tasks[: args.limit] if args.limit else tasks
    cfg = yaml.safe_load(open(args.config)) if os.path.exists(args.config) else {"systems": {}}
    system = load_system(cfg, args.system, args.mode)
    registry = benchmark_registry()
    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"{args.system}__{args.mode}"
    pred_path = os.path.join(args.out_dir, f"predictions__{tag}.jsonl")
    outputs = asyncio.run(run_all(tasks, args.mode, system, registry, args.concurrency, pred_path))
    report = score_predictions(tasks, outputs)
    with open(os.path.join(args.out_dir, f"metrics__{tag}.json"), "w") as fh:
        json.dump({"system": args.system, "mode": args.mode, "n_tasks": len(tasks), "metrics": report}, fh, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

