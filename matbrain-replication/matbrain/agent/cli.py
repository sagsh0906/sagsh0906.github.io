"""Run MatBrain on a query.

    matbrain --config configs/matbrain.yaml "Provide the CIF file of CsPb(Cl0.2Br0.4I0.4)3 perovskite."

Models are served with vLLM (see deploy/serve_models.sh); tools run in-process
by default or on a remote Mat-MCP server (``tools.mcp_url`` in the config).
"""

from __future__ import annotations

import argparse
import asyncio
import json

import yaml

from matbrain.agent.executor import RemoteMCPTools
from matbrain.agent.graph import MatBrain
from matbrain.agent.llm import backend_from_config
from matbrain.mcp import load_registry


async def _run(cfg: dict, query: str) -> dict:
    registry = load_registry()
    tcfg = cfg.get("tools", {})
    registry = registry.subset(include=tcfg.get("include"), exclude=tcfg.get("exclude"), disable_target_db=tcfg.get("disable_target_db", False))
    kwargs = dict(
        executor_backend=backend_from_config(cfg["mat_t1"]),
        reasoner_backend=backend_from_config(cfg["mat_r1"]),
        registry=registry,
        max_iterations=cfg.get("max_iterations", 6),
        max_tool_calls=cfg.get("max_tool_calls", 12),
        start_with_analysis=cfg.get("start_with_analysis", False),
    )
    if tcfg.get("mcp_url"):
        async with RemoteMCPTools(tcfg["mcp_url"], registry) as remote:
            result = await MatBrain(dispatcher=remote, **kwargs).arun(query)
    else:
        result = await MatBrain(**kwargs).arun(query)
    return result.__dict__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--config", default="configs/matbrain.yaml")
    ap.add_argument("--trace-out", help="write the full state (executions, analyses) as JSON")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = asyncio.run(_run(cfg, args.query))
    print(out["answer"])
    if args.trace_out:
        with open(args.trace_out, "w") as fh:
            json.dump(out, fh, indent=2, default=str, ensure_ascii=False)


if __name__ == "__main__":
    main()
