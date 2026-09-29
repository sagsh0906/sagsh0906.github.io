"""Retrieve MP-grounded records (structure CIF + property JSON + DOIs).

    MP_API_KEY=... python -m matbrain.data.mp_fetch --out data/mp/mp_records.jsonl

Each record links structure (CIF), computed properties (JSON fields) and
literature (DOIs parsed from the provenance BibTeX references), the
identifiers used to align database entries with literature in the Mat-SFT
corpus. ``created_at`` drives the timestamp-based train/val/test split.
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

from matbrain.chem import structure_to_cif
from matbrain.data.formats import write_jsonl

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "symmetry",
    "nsites",
    "energy_per_atom",
    "uncorrected_energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "efermi",
    "band_gap",
    "is_metal",
    "is_magnetic",
    "ordering",
    "total_magnetization",
    "theoretical",
    "database_IDs",
]
DOI_RE = re.compile(r"doi\s*=\s*[{\"]\s*([^}\"]+?)\s*[}\"]", re.I)


def dois_from_bibtex(refs: list[str] | None) -> list[str]:
    out = []
    for ref in refs or []:
        out += [d.strip() for d in DOI_RE.findall(ref)]
    return sorted(set(out))


def _dump(doc: Any) -> dict:
    return doc if isinstance(doc, dict) else doc.model_dump()


def summary_to_record(d: dict, created_at: Any, dois: list[str]) -> dict:
    from pymatgen.core import Structure

    s = d["structure"] if isinstance(d["structure"], Structure) else Structure.from_dict(d["structure"])
    sym = d.get("symmetry") or {}
    ordering = d.get("ordering")
    return {
        "material_id": str(d["material_id"]),
        "formula": d.get("formula_pretty"),
        "cif": structure_to_cif(s),
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
        "nsites": d.get("nsites"),
        "spacegroup_symbol": sym.get("symbol"),
        "spacegroup_number": sym.get("number"),
        "crystal_system": str(sym.get("crystal_system")).lower() if sym.get("crystal_system") else None,
        "full_formula": s.composition.formula.replace(" ", ""),
        "energy_per_atom": d.get("energy_per_atom"),
        "uncorrected_energy_per_atom": d.get("uncorrected_energy_per_atom"),
        "formation_energy_per_atom": d.get("formation_energy_per_atom"),
        "energy_above_hull": d.get("energy_above_hull"),
        "efermi": d.get("efermi"),
        "band_gap": d.get("band_gap"),
        "is_metal": d.get("is_metal"),
        "is_magnetic": d.get("is_magnetic"),
        "ordering": getattr(ordering, "value", ordering),
        "total_magnetization": d.get("total_magnetization"),
        "theoretical": d.get("theoretical"),
        "icsd_ids": (d.get("database_IDs") or {}).get("icsd", []),
        "dois": dois,
    }


def fetch(api_key: str, limit: int | None = None) -> list[dict]:
    from mp_api.client import MPRester

    with MPRester(api_key) as mpr:
        kw = {"num_chunks": max(1, limit // 1000), "chunk_size": min(limit, 1000)} if limit else {}
        summaries = [_dump(d) for d in mpr.materials.summary.search(fields=SUMMARY_FIELDS, **kw)]
        ids = [str(d["material_id"]) for d in summaries]
        created = {str(_dump(d)["material_id"]): _dump(d).get("created_at") for d in mpr.materials.search(material_ids=ids, fields=["material_id", "created_at"])}
        prov = {}
        for d in mpr.materials.provenance.search(material_ids=ids, fields=["material_id", "references", "created_at"]):
            d = _dump(d)
            prov[str(d["material_id"])] = d
    records = []
    for d in summaries:
        mid = str(d["material_id"])
        p = prov.get(mid, {})
        records.append(summary_to_record(d, created.get(mid) or p.get("created_at"), dois_from_bibtex(p.get("references"))))
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/mp/mp_records.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise SystemExit("set MP_API_KEY")
    n = write_jsonl(args.out, fetch(key, args.limit))
    print(f"wrote {n} records to {args.out}")


if __name__ == "__main__":
    main()
