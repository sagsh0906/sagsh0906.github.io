"""Build a leakage-safe reference-entry snapshot for the PhaseDiagram tool.

Only training-split MP records are used, so held-out materials can never sit
on the reference convex hull during benchmarking:

    python scripts/build_reference_entries.py --records data/splits/train.jsonl --out data/reference_entries.jsonl
    export MATBRAIN_REFERENCE_ENTRIES=$PWD/data/reference_entries.jsonl

Energies are MP-corrected total energies per atom (GGA/GGA+U mixing), so ML
candidate energies must be MP2020-corrected as well (phase_diagram_ehull does
this by default).
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from matbrain.data.formats import iter_records, write_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rows = []
    for r in iter_records(args.records):
        if r.get("energy_per_atom") is None:
            continue
        rows.append({"material_id": r["material_id"], "composition": r.get("full_formula") or r["formula"], "energy_per_atom": r["energy_per_atom"]})
    n = write_jsonl(args.out, rows)
    print(f"wrote {n} reference entries to {args.out}")


if __name__ == "__main__":
    main()
