"""Leakage-controlled benchmark tasks (paper Fig. 3).

Three task families built from the clean held-out MP split:
  * property classification: metallicity, magnetism (balanced accuracy),
    magnetic ordering NM/FM/FiM/AFM (macro F1);
  * property regression: formation energy, energy above hull, Fermi energy (MAE);
  * structure design: natural-language target (composition, atom count,
    crystal system, space group) -> CIF, scored with seven metrics.

Inputs are a held-out CIF + task prompt (property tasks) or a natural-language
target (design). No MP/OQMD/ICSD identifiers are given to the models.

    python -m matbrain.data.benchmark --clean data/audit/test_clean.jsonl --out data/benchmark/tasks.jsonl
"""

from __future__ import annotations

import argparse
import random
from typing import Any

from pymatgen.core import Composition

from matbrain.data.formats import iter_records, write_jsonl

CLASSIFICATION = {
    "is_metal": ("Is this material metallic (zero band gap in DFT-PBE)?", ["true", "false"]),
    "is_magnetic": ("Is this material magnetic (non-zero magnetic moments in the DFT ground state)?", ["true", "false"]),
    "ordering": ("What is the collinear magnetic ordering of this material?", ["NM", "FM", "FiM", "AFM"]),
}
REGRESSION = {
    "formation_energy_per_atom": ("formation energy per atom", "eV/atom"),
    "energy_above_hull": ("energy above the convex hull", "eV/atom"),
    "efermi": ("Fermi energy", "eV"),
}

PROPERTY_TEMPLATE = (
    "Below is the crystal structure of a material in CIF format.\n\n{cif}\n\n{question}\n"
    "Give your final answer inside <answer></answer> as JSON: {answer_format}"
)
DESIGN_TEMPLATE = (
    "Design a crystal structure with the following target specification:\n"
    "- composition (unit cell): {formula}\n"
    "- number of atoms in the unit cell: {nsites}\n"
    "- crystal system: {crystal_system}\n"
    "- space group: {spacegroup_symbol} (No. {spacegroup_number})\n"
    "The structure must be chemically reasonable (charge balanced) and geometrically valid. "
    "Give the final structure inside <answer></answer> as a complete CIF."
)


def _norm_label(prop: str, value: Any) -> Any:
    if prop in ("is_metal", "is_magnetic"):
        return bool(value)
    if prop == "ordering":
        canon = {"NM": "NM", "FM": "FM", "FIM": "FiM", "AFM": "AFM"}
        return canon.get(str(value).upper()) if value is not None else None  # drops "Unknown"
    return float(value) if value is not None else None


def build_tasks(records: list[dict], properties: list[str] | None = None, design: bool = True) -> list[dict[str, Any]]:
    from matbrain.chem import parse_structure

    tasks = []
    props = properties or list(CLASSIFICATION) + list(REGRESSION)
    for rec in records:
        mid = rec["material_id"]
        flags = {"formula_overlap": bool(rec.get("formula_overlap_flag", False))}
        for prop in props:
            label = _norm_label(prop, rec.get(prop))
            if label is None:
                continue
            if prop in CLASSIFICATION:
                q, choices = CLASSIFICATION[prop]
                fmt = '{"value": ' + " | ".join(f'"{c}"' if prop == "ordering" else c for c in choices) + "}"
                family = "classification"
            else:
                name, unit = REGRESSION[prop]
                q = f"Predict the DFT (PBE/PBE+U, Materials Project compatible) {name} of this material in {unit}."
                fmt = '{"value": <number>}'
                family = "regression"
            tasks.append({
                "id": f"{mid}:{prop}", "source_id": mid, "family": family, "property": prop,
                "prompt": PROPERTY_TEMPLATE.format(cif=rec["cif"].strip(), question=q, answer_format=fmt),
                "input_cif": rec["cif"], "label": label, "flags": flags,
            })
        if design:
            s = parse_structure(rec["cif"])
            target = {
                "formula": Composition(s.composition.formula).formula.replace(" ", ""),
                "reduced_formula": s.composition.reduced_formula,
                "nsites": len(s),
                "crystal_system": rec.get("crystal_system") or "",
                "spacegroup_symbol": rec.get("spacegroup_symbol"),
                "spacegroup_number": rec.get("spacegroup_number"),
            }
            if target["spacegroup_number"]:
                if not target["crystal_system"]:
                    from matbrain.chem import crystal_system_from_number

                    target["crystal_system"] = crystal_system_from_number(int(target["spacegroup_number"]))
                tasks.append({
                    "id": f"{mid}:design", "source_id": mid, "family": "structure_design", "property": "structure",
                    "prompt": DESIGN_TEMPLATE.format(**target), "target": target, "label": None, "flags": flags,
                })
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", required=True)
    ap.add_argument("--out", default="data/benchmark/tasks.jsonl")
    ap.add_argument("--sample", type=int, default=None, help="subsample source records")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    records = list(iter_records(args.clean))
    if args.sample:
        records = random.Random(args.seed).sample(records, min(args.sample, len(records)))
    tasks = build_tasks(records)
    n = write_jsonl(args.out, tasks)
    print(f"{n} tasks from {len(records)} held-out records -> {args.out}")


if __name__ == "__main__":
    main()
