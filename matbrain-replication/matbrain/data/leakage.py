"""Five-level leakage audit and leakage-controlled benchmark construction.

Levels (paper Methods / Supplementary Table 13):
  1. strict MP-ID overlap                                   -> excluded
  2. exact CIF-hash overlap                                 -> excluded
  3. raw-formula (reduced composition) overlap              -> flag only
  4. full AFLOW-style protostructure-label overlap          -> excluded
  5. structural-fingerprint neighbours confirmed by
     pymatgen StructureMatcher                              -> excluded

    python -m matbrain.data.leakage --train data/splits/train.jsonl \
        --test data/splits/test.jsonl --out-dir data/audit

Outputs: per-sample flags (jsonl), a summary table (csv + markdown) and the
clean held-out split (test_clean.jsonl; 1,662 of 2,000 in the paper).
"""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

from matbrain import chem
from matbrain.data.formats import iter_records, write_jsonl

EXCLUDING_LEVELS = ("mp_id", "cif_hash", "protostructure", "structure_match")
N_ELEMENTS = 103


@dataclass
class AuditConfig:
    symprec: float = 0.1
    k_neighbours: int = 10
    ltol: float = 0.2
    stol: float = 0.3
    angle_tol: float = 5.0
    workers: int = max(1, (os.cpu_count() or 2) - 1)


@dataclass
class SampleFlags:
    material_id: str
    mp_id: bool = False
    cif_hash: bool = False
    formula: bool = False
    protostructure: bool = False
    structure_match: bool = False
    matched_train_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def excluded(self) -> bool:
        return any(getattr(self, lvl) for lvl in EXCLUDING_LEVELS)


# --------------------------------------------------------------------------- #
# Per-record descriptors (parallelised: symmetry analysis dominates runtime)
# --------------------------------------------------------------------------- #
def fingerprint(structure) -> np.ndarray:
    """Composition + lattice-shape + symmetry fingerprint for neighbour retrieval."""
    comp = np.zeros(N_ELEMENTS)
    for el, frac in structure.composition.fractional_composition.items():
        if el.Z <= N_ELEMENTS:
            comp[el.Z - 1] = frac
    a, b, c = sorted(structure.lattice.abc)
    angles = np.array(structure.lattice.angles) / 180.0
    lat = np.array([np.cbrt(structure.volume / len(structure)) / 5.0, a / c, b / c, *angles])
    return np.concatenate([comp * 3.0, lat])


def describe(rec: dict[str, Any], symprec: float = 0.1) -> dict[str, Any]:
    out: dict[str, Any] = {"material_id": rec["material_id"], "errors": []}
    cif = rec.get("cif", "")
    out["cif_hash"] = chem.cif_hash(cif) if cif else None
    try:
        s = chem.parse_structure(cif)
    except chem.StructureParseError as exc:
        out["errors"].append(str(exc))
        return out
    out["formula"] = s.composition.reduced_formula
    out["fingerprint"] = fingerprint(s)
    try:
        out["protostructure"] = chem.protostructure_label(s, symprec=symprec)
    except Exception as exc:
        out["errors"].append(f"protostructure: {exc}")
    return out


def _describe_all(records: list[dict], cfg: AuditConfig) -> list[dict]:
    if cfg.workers <= 1 or len(records) < 64:
        return [describe(r, cfg.symprec) for r in records]
    with ProcessPoolExecutor(cfg.workers) as pool:
        return list(pool.map(describe, records, [cfg.symprec] * len(records), chunksize=64))


# --------------------------------------------------------------------------- #
def _nearest(train_fp: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    d = np.linalg.norm(train_fp - query[None, :], axis=1)
    k = min(k, len(d))
    idx = np.argpartition(d, k - 1)[:k]
    return idx[np.argsort(d[idx])]


def audit(train: list[dict], test: list[dict], cfg: AuditConfig | None = None, progress: bool = False) -> list[SampleFlags]:
    from pymatgen.analysis.structure_matcher import StructureMatcher

    cfg = cfg or AuditConfig()
    tr = _describe_all(train, cfg)
    te = _describe_all(test, cfg)
    train_ids = {r["material_id"] for r in train}
    train_hashes = {d["cif_hash"] for d in tr if d.get("cif_hash")}
    train_formulas = {d["formula"] for d in tr if d.get("formula")}
    train_protos = {d["protostructure"] for d in tr if d.get("protostructure")}
    fp_rows = [(i, d["fingerprint"]) for i, d in enumerate(tr) if "fingerprint" in d]
    fp_index = np.array([i for i, _ in fp_rows])
    fp_matrix = np.stack([f for _, f in fp_rows]) if fp_rows else np.zeros((0, N_ELEMENTS + 6))
    matcher = StructureMatcher(ltol=cfg.ltol, stol=cfg.stol, angle_tol=cfg.angle_tol)
    train_by_id = {r["material_id"]: r for r in train}

    flags = []
    for n, (rec, d) in enumerate(zip(test, te)):
        f = SampleFlags(material_id=rec["material_id"], errors=list(d["errors"]))
        f.mp_id = rec["material_id"] in train_ids
        f.cif_hash = bool(d.get("cif_hash")) and d["cif_hash"] in train_hashes
        f.formula = d.get("formula") in train_formulas
        f.protostructure = d.get("protostructure") in train_protos
        if "fingerprint" in d and len(fp_matrix):
            s = chem.parse_structure(rec["cif"])
            for j in _nearest(fp_matrix, d["fingerprint"], cfg.k_neighbours):
                cand = train_by_id[tr[fp_index[j]]["material_id"]]
                try:
                    if matcher.fit(s, chem.parse_structure(cand["cif"])):
                        f.structure_match = True
                        f.matched_train_ids.append(cand["material_id"])
                except Exception as exc:
                    f.errors.append(f"structure_matcher: {exc}")
        flags.append(f)
        if progress and n % 100 == 0:
            print(f"audited {n}/{len(test)}", flush=True)
    return flags


def summarize(flags: Iterable[SampleFlags]) -> list[dict[str, Any]]:
    flags = list(flags)
    n = len(flags)
    rows = []
    for level, label, action in [
        ("mp_id", "MP-ID overlap", "excluded"),
        ("cif_hash", "Exact CIF-hash overlap", "excluded"),
        ("formula", "Raw-formula composition overlap", "flag only"),
        ("protostructure", "Full AFLOW protostructure-label overlap", "excluded"),
        ("structure_match", "StructureMatcher-confirmed similarity", "excluded"),
    ]:
        k = sum(getattr(f, level) for f in flags)
        rows.append({"level": label, "n_overlap": k, "fraction": round(k / n, 4) if n else 0.0, "action": action})
    excl = sum(f.excluded for f in flags)
    rows.append({"level": "Excluded (union of excluding levels)", "n_overlap": excl, "fraction": round(excl / n, 4) if n else 0.0, "action": f"clean = {n - excl}"})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out-dir", default="data/audit")
    ap.add_argument("--workers", type=int, default=AuditConfig.workers)
    args = ap.parse_args()
    train, test = list(iter_records(args.train)), list(iter_records(args.test))
    flags = audit(train, test, AuditConfig(workers=args.workers), progress=True)
    os.makedirs(args.out_dir, exist_ok=True)
    write_jsonl(os.path.join(args.out_dir, "audit_flags.jsonl"), [{**asdict(f), "excluded": f.excluded} for f in flags])
    rows = summarize(flags)
    with open(os.path.join(args.out_dir, "audit_summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out_dir, "audit_summary.md"), "w") as fh:
        fh.write("| Level | Overlap | Fraction | Action |\n|---|---|---|---|\n")
        for r in rows:
            fh.write(f"| {r['level']} | {r['n_overlap']} | {r['fraction']} | {r['action']} |\n")
    excluded = {f.material_id for f in flags if f.excluded}
    formula_flag = {f.material_id for f in flags if f.formula}
    clean = [{**r, "formula_overlap_flag": r["material_id"] in formula_flag} for r in test if r["material_id"] not in excluded]
    write_jsonl(os.path.join(args.out_dir, "test_clean.jsonl"), clean)
    for r in rows:
        print(r)
    print(f"clean held-out examples: {len(clean)} / {len(test)}")


if __name__ == "__main__":
    main()
