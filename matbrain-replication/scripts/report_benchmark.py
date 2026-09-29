"""Aggregate metrics__*.json files into markdown tables (cf. Fig. 3 / Supplementary Tables 5-7).

    python scripts/report_benchmark.py --dir results/benchmark > results/benchmark/report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os

CLS = [("is_metal", "balanced_accuracy", "Metallic (B.acc)"), ("is_magnetic", "balanced_accuracy", "Magnetic (B.acc)"), ("ordering", "macro_f1", "Magnetic order (F1)")]
REG = [("formation_energy_per_atom", "Formation energy MAE"), ("energy_above_hull", "E_hull MAE"), ("efermi", "Fermi energy MAE")]
DES = ["complete_success", "valid_structure", "charge_balanced", "formula", "atom_count", "crystal_system", "space_group"]


def fmt(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/benchmark")
    args = ap.parse_args()
    runs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(args.dir, "metrics__*.json")))]
    print("## Property classification\n\n| System | Mode | " + " | ".join(c[2] for c in CLS) + " |\n|---|---|" + "---|" * len(CLS))
    for r in runs:
        m = r["metrics"]
        print(f"| {r['system']} | {r['mode']} | " + " | ".join(fmt(m.get(p, {}).get(k)) for p, k, _ in CLS) + " |")
    print("\n## Property regression (MAE, coverage)\n\n| System | Mode | " + " | ".join(c[1] for c in REG) + " |\n|---|---|" + "---|" * len(REG))
    for r in runs:
        m = r["metrics"]
        cells = [f"{fmt(m.get(p, {}).get('mae'))} ({fmt(m.get(p, {}).get('coverage'))})" for p, _ in REG]
        print(f"| {r['system']} | {r['mode']} | " + " | ".join(cells) + " |")
    print("\n## Structure design\n\n| System | Mode | " + " | ".join(DES) + " |\n|---|---|" + "---|" * len(DES))
    for r in runs:
        d = r["metrics"].get("structure_design", {})
        print(f"| {r['system']} | {r['mode']} | " + " | ".join(fmt(d.get(k)) for k in DES) + " |")


if __name__ == "__main__":
    main()
