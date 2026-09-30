"""Create small synthetic MP-like records for smoke-testing the pipeline offline.

The structures are real prototypes built with pymatgen; the property values
and timestamps are synthetic placeholders (NOT Materials Project data).

    python scripts/make_toy_records.py --out examples/data/toy_records.jsonl
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pymatgen.core import Lattice, Structure  # noqa: E402

from matbrain.chem import structure_to_cif, symmetry_info  # noqa: E402
from matbrain.data.formats import write_jsonl  # noqa: E402

PROTOTYPES = [
    ("Fm-3m", ["A", "B"], [[0, 0, 0], [0.5, 0.5, 0.5]], [("Na", "Cl", 5.64), ("K", "Cl", 6.29), ("Mg", "O", 4.21), ("Li", "F", 4.03), ("Ca", "O", 4.81), ("Ni", "O", 4.18), ("Na", "F", 4.63), ("K", "Br", 6.60)]),
    ("Pm-3m", ["A", "B"], [[0, 0, 0], [0.5, 0.5, 0.5]], [("Cs", "Cl", 4.12), ("Cs", "Br", 4.29), ("Cs", "I", 4.57), ("Tl", "Cl", 3.84)]),
    ("F-43m", ["A", "B"], [[0, 0, 0], [0.25, 0.25, 0.25]], [("Zn", "S", 5.41), ("Ga", "As", 5.65), ("In", "P", 5.87), ("Cd", "Te", 6.48)]),
    ("Fm-3m", ["A", "B"], [[0, 0, 0], [0.25, 0.25, 0.25]], [("Ca", "F", 5.46), ("Sr", "F", 5.80), ("Ba", "F", 6.20), ("Ce", "O", 5.41)]),
]
PEROVSKITES = [("Sr", "Ti", "O", 3.905), ("Ba", "Ti", "O", 4.00), ("Ca", "Ti", "O", 3.84), ("K", "Nb", "O", 4.00), ("Cs", "Pb", "Br", 5.95), ("Cs", "Pb", "Cl", 5.68)]


def build() -> list[Structure]:
    out = []
    for sg, _, coords, members in PROTOTYPES:
        for a, b, lat in members:  # rocksalt / CsCl / zincblende AB, fluorite AB2 (8c)
            out.append(Structure.from_spacegroup(sg, Lattice.cubic(lat), [a, b], coords))
    for a, b, x, lat in PEROVSKITES:
        out.append(Structure.from_spacegroup("Pm-3m", Lattice.cubic(lat), [a, b, x], [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0]]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/data/toy_records.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    t0 = datetime(2018, 1, 1)
    rows = []
    for i, s in enumerate(build()):
        sym = symmetry_info(s)
        magnetic = "Ni" in s.composition
        rows.append({
            "material_id": f"toy-{i + 1}",
            "formula": s.composition.reduced_formula,
            "full_formula": s.composition.formula.replace(" ", ""),
            "cif": structure_to_cif(s),
            "created_at": (t0 + timedelta(days=30 * i)).isoformat(),
            "nsites": len(s),
            "spacegroup_symbol": sym["space_group_symbol"],
            "spacegroup_number": sym["space_group_number"],
            "crystal_system": sym["crystal_system"],
            "energy_per_atom": round(rng.uniform(-8, -3), 4),
            "formation_energy_per_atom": round(rng.uniform(-3.5, -0.5), 4),
            "energy_above_hull": round(abs(rng.gauss(0, 0.05)), 4),
            "efermi": round(rng.uniform(0, 6), 4),
            "band_gap": round(rng.uniform(0, 6), 3),
            "is_metal": False,
            "is_magnetic": magnetic,
            "ordering": "AFM" if magnetic else "NM",
            "synthetic": True,
        })
    n = write_jsonl(args.out, rows)
    print(f"wrote {n} synthetic records to {args.out}")


if __name__ == "__main__":
    main()
