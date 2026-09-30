"""Smoke-test Mat-MCP tool backends before spending GPU hours on RL / evaluation.

    python scripts/check_tools.py                                   # in-process registry
    python scripts/check_tools.py --mcp-url http://127.0.0.1:8000/mcp   # a running Mat-MCP server

Each tool is called once on rocksalt NaCl (a few seconds each on GPU; the first
MatGL call also downloads model weights). Optional backends that are not
configured (CrystaLLM, MP API, reference entries) are reported as SKIP.
Exit code 1 if any required tool fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymatgen.core import Lattice, Structure  # noqa: E402

from matbrain.chem import structure_to_cif  # noqa: E402
from matbrain.mcp import load_registry  # noqa: E402

NACL = structure_to_cif(Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]))

# (tool, arguments, required?, skip-unless-env)
CHECKS = [
    ("validate_structure", {"cif": NACL}, True, None),
    ("analyze_symmetry", {"cif": NACL}, True, None),
    ("check_charge_balance", {"formula": "CoV4S8"}, True, None),
    ("predict_energy", {"cif": NACL, "model": "CHGNet"}, True, None),
    ("relax_structure", {"cif": NACL, "model": "CHGNet", "steps": 50}, True, None),
    ("predict_formation_energy", {"cif": NACL}, True, None),
    ("predict_bandgap", {"cif": NACL}, True, None),
    ("predict_magnetic_moments", {"cif": NACL}, True, None),
    ("phase_diagram_ehull", {"cif": NACL, "relax": False}, False, "MATBRAIN_REFERENCE_ENTRIES"),
    ("crystallm_generate", {"formula": "Na4Cl4", "spacegroup": "Fm-3m", "num_samples": 1}, False, "CRYSTALLM_HOME"),
    ("mp_search", {"formula": "NaCl", "max_results": 1}, False, "MP_API_KEY"),
]


async def run(dispatcher, registry) -> int:
    failed = 0
    for name, args, required, env in CHECKS:
        if name not in registry:
            print(f"SKIP  {name:<26} not in this tool pool")
            continue
        if env and not os.environ.get(env):
            print(f"SKIP  {name:<26} set {env} to test it")
            continue
        t0 = time.perf_counter()
        res = await dispatcher.acall(name, args, strict_handles=True)
        dt = time.perf_counter() - t0
        if res.ok:
            preview = json.dumps(res.data, default=str, ensure_ascii=False)[:90]
            print(f"OK    {name:<26} {dt:6.1f}s  {preview}")
        else:
            failed += required
            print(f"{'FAIL' if required else 'WARN'}  {name:<26} {dt:6.1f}s  {res.error}")
    return failed


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mcp-url")
    args = ap.parse_args()
    registry = load_registry()
    if args.mcp_url:
        from matbrain.agent.executor import RemoteMCPTools

        async with RemoteMCPTools(args.mcp_url, registry) as remote:
            failed = await run(remote, registry)
    else:
        failed = await run(registry, registry)
    print(f"\n{failed} required tool(s) failed" if failed else "\nall required tools OK")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    asyncio.run(main())
