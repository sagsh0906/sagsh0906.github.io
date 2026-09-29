"""Generate verl tool-config YAML files for Mat-T1 RL from the Mat-MCP registry.

    python training/rl/make_tool_config.py --mode local  --out training/rl/tool_config/mat_mcp_local.yaml
    python training/rl/make_tool_config.py --mode remote --out training/rl/tool_config/mat_mcp_remote.yaml \
        --mcp-url http://mat-mcp:8000/mcp

Mat-20K-RL tasks cover structure generation and property prediction, so the
default pool excludes nothing; pass --disable-target-db to reproduce the
"no shortcut via a single DB lookup" setting (the paper states the tasks are
designed so the agent cannot maximise reward by one trivial API lookup).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from matbrain.mcp import load_registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["local", "remote"], default="local")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mcp-url", default="http://127.0.0.1:8000/mcp")
    ap.add_argument("--disable-target-db", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=["vasp_prepare_inputs", "mattergen_generate"])
    ap.add_argument("--max-observation-chars", type=int, default=4000)
    args = ap.parse_args()

    registry = load_registry().subset(exclude=args.exclude, disable_target_db=args.disable_target_db)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.mode == "local":
        tools = [
            {
                "class_name": "matbrain.rl.verl_tools.MatMCPNativeTool",
                "config": {"type": "native", "max_observation_chars": args.max_observation_chars},
                "tool_schema": spec.openai_schema(flat=True),
            }
            for spec in registry
        ]
        cfg = {"tools": tools}
    else:
        servers_path = os.path.join(os.path.dirname(os.path.abspath(args.out)), "mcp_servers.json")
        with open(servers_path, "w") as fh:
            json.dump({"mcpServers": {"mat-mcp": {"url": args.mcp_url}}}, fh, indent=2)
        cfg = {
            "tools": [
                {
                    "class_name": "verl.tools.mcp_base_tool.MCPBaseTool",
                    "config": {"type": "mcp", "rate_limit": 256, "timeout": 600},
                    "mcp": {"mcp_servers_config_path": servers_path, "tool_selected_list": registry.names()},
                }
            ]
        }
    with open(args.out, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True, width=120)
    print(f"wrote {len(registry)} tools to {args.out}")


if __name__ == "__main__":
    main()
