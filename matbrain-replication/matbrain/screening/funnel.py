"""Seven-step high-throughput screening funnel for M-V-S (M = Fe, Co, Ni) NRR
catalysts (paper Fig. 6b, Supplementary Table 9).

  01 Component           strict ternary M-V-S composition          30,000 -> 26,973
  02 Valence state       charge-neutral oxidation states           -> 21,954
  03 Deduplication       StructureMatcher (stol 0.3, ltol 0.2, 5 deg) -> 10,452
  04 Structure           geometric validity (no atomic overlaps)   -> 10,128
  05 Stability           E_hull <= 0.025 eV/atom (PhaseDiagram)    -> 42
  06 Electronic property MEGNet band gap, semiconductor < 2.0 eV   -> 42
  07 Novelty             no StructureMatcher match in databases    -> 38
then ranking by V/(V+M), stability and band gap (CoV4S8 first in the paper).

The expensive property evaluations are injected as callables so the funnel can
run with MatGL/CHGNet + MP references (default) or any other backend.

    python -m matbrain.screening.funnel --handles-file mattergen/handles.txt --out-dir results/nrr_screening
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Sequence

from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure

from matbrain import chem


@dataclass
class FunnelConfig:
    metals: tuple[str, ...] = ("Fe", "Co", "Ni")
    framework: tuple[str, ...] = ("V", "S")
    stol: float = 0.3
    ltol: float = 0.2
    angle_tol: float = 5.0
    min_distance: float = chem.MIN_BOND_DISTANCE
    ehull_max: float = 0.025
    gap_min: float = 0.0  # exclusive: semiconductors only
    gap_max: float = 2.0
    rank_element: str = "V"


@dataclass
class Stage:
    name: str
    n_in: int
    n_out: int
    seconds: float

    @property
    def pass_rate(self) -> float:
        return self.n_out / self.n_in if self.n_in else 0.0


@dataclass
class Candidate:
    structure: Structure
    id: str
    props: dict[str, Any] = field(default_factory=dict)


class ScreeningFunnel:
    def __init__(
        self,
        cfg: FunnelConfig | None = None,
        ehull_fn: Callable[[list[Structure]], list[float | None]] | None = None,
        bandgap_fn: Callable[[Structure], float | None] | None = None,
        reference_fn: Callable[[str], list[Structure]] | None = None,
    ):
        self.cfg = cfg or FunnelConfig()
        self.ehull_fn = ehull_fn or default_ehull_fn
        self.bandgap_fn = bandgap_fn or default_bandgap_fn
        self.reference_fn = reference_fn or default_reference_fn
        self.stages: list[Stage] = []

    def _stage(self, name: str, cands: list[Candidate], keep: Callable[[list[Candidate]], list[Candidate]]) -> list[Candidate]:
        t0 = time.perf_counter()
        out = keep(cands)
        self.stages.append(Stage(name, len(cands), len(out), time.perf_counter() - t0))
        return out

    # ----------------------------------------------------------------- steps
    def component(self, cands):
        cfg = self.cfg
        out = []
        for c in cands:
            els = {e.symbol for e in c.structure.composition.elements}
            metals = els - set(cfg.framework)
            if set(cfg.framework) <= els and len(els) == len(cfg.framework) + 1 and len(metals) == 1 and metals <= set(cfg.metals):
                out.append(c)
        return out

    def valence(self, cands):
        return [c for c in cands if chem.charge_balance(c.structure.composition)["charge_balanced"]]

    def deduplicate(self, cands):
        sm = StructureMatcher(ltol=self.cfg.ltol, stol=self.cfg.stol, angle_tol=self.cfg.angle_tol)
        by_formula: dict[str, list[Candidate]] = {}
        for c in cands:
            by_formula.setdefault(c.structure.composition.reduced_formula, []).append(c)
        out = []
        for group in by_formula.values():
            reps: list[Candidate] = []
            for c in group:
                if not any(sm.fit(c.structure, r.structure) for r in reps):
                    reps.append(c)
            out.extend(reps)
        return out

    def geometry(self, cands):
        return [c for c in cands if chem.structural_validity(c.structure, self.cfg.min_distance)["valid"]]

    def stability(self, cands):
        values = self.ehull_fn([c.structure for c in cands])
        out = []
        for c, v in zip(cands, values):
            c.props["e_above_hull"] = v
            if v is not None and v <= self.cfg.ehull_max:
                out.append(c)
        return out

    def electronic(self, cands):
        out = []
        for c in cands:
            gap = self.bandgap_fn(c.structure)
            c.props["band_gap"] = gap
            if gap is not None and self.cfg.gap_min < gap < self.cfg.gap_max:
                out.append(c)
        return out

    def novelty(self, cands):
        sm = StructureMatcher(ltol=self.cfg.ltol, stol=self.cfg.stol, angle_tol=self.cfg.angle_tol)
        cache: dict[str, list[Structure]] = {}
        out = []
        for c in cands:
            key = chem.chemical_system(c.structure)
            if key not in cache:
                cache[key] = self.reference_fn(key)
            known = [r for r in cache[key] if r.composition.reduced_formula == c.structure.composition.reduced_formula]
            c.props["novel"] = not any(sm.fit(c.structure, r) for r in known)
            if c.props["novel"]:
                out.append(c)
        return out

    # --------------------------------------------------------------- driver
    def run(self, structures: Sequence[Structure], ids: Sequence[str] | None = None) -> list[Candidate]:
        self.stages = []
        cands = [Candidate(s, ids[i] if ids else str(i)) for i, s in enumerate(structures)]
        for name, fn in [
            ("01 Component", self.component),
            ("02 Valence state", self.valence),
            ("03 Deduplication", self.deduplicate),
            ("04 Structure", self.geometry),
            ("05 Stability (Ehull)", self.stability),
            ("06 Electronic property", self.electronic),
            ("07 Novelty (against DB)", self.novelty),
        ]:
            cands = self._stage(name, cands, fn)
        return self.rank(cands)

    def rank(self, cands: list[Candidate]) -> list[Candidate]:
        """V/(V+M) descending, then E_hull ascending, then distance of the gap from 1 eV."""
        el = self.cfg.rank_element
        for c in cands:
            metals = [e.symbol for e in c.structure.composition.elements if e.symbol not in ("S",)]
            c.props["formula"] = c.structure.composition.reduced_formula
            c.props[f"{el}_fraction"] = round(chem.element_fraction(c.structure.composition, el, metals), 4)
            c.props["space_group"] = chem.symmetry_info(c.structure)["space_group_symbol"]
        return sorted(cands, key=lambda c: (-c.props[f"{el}_fraction"], c.props.get("e_above_hull") or 0.0, abs((c.props.get("band_gap") or 1.0) - 1.0)))

    def report(self) -> list[dict[str, Any]]:
        return [{**asdict(s), "pass_rate": round(s.pass_rate, 4)} for s in self.stages]


# --------------------------------------------------------------------------- #
# Default backends (MatGL + MP / local references)
# --------------------------------------------------------------------------- #
def default_ehull_fn(structures: list[Structure]) -> list[float | None]:
    from matbrain.mcp.tools.phase import candidate_entry, ehull_batch

    entries = [candidate_entry(s, energy_model="CHGNet", relax=True) for s in structures]
    return [r.get("energy_above_hull") for r in ehull_batch(entries)]


def default_bandgap_fn(structure: Structure) -> float | None:
    import torch

    from matbrain.mcp.tools.properties import BANDGAP_FIDELITY, load_matgl_model

    model = load_matgl_model("bandgap", "MEGNet")
    return max(0.0, float(model.predict_structure(structure=structure, state_attr=torch.tensor([BANDGAP_FIDELITY["PBE"]]))))


def default_reference_fn(chemsys: str) -> list[Structure]:
    path = os.environ.get("MATBRAIN_NOVELTY_DB")
    if path:
        from matbrain.data.formats import iter_records

        return [chem.parse_structure(r["cif"]) for r in iter_records(path) if chem.chemical_system(r.get("formula") or chem.parse_structure(r["cif"])) == chemsys]
    from mp_api.client import MPRester

    with MPRester(os.environ["MP_API_KEY"]) as mpr:
        return [d.structure for d in mpr.materials.summary.search(chemsys=chemsys, fields=["structure"])]


def main() -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--handles-file", help="artifact handles written by mattergen_generate")
    src.add_argument("--cif-dir")
    ap.add_argument("--out-dir", default="results/nrr_screening")
    args = ap.parse_args()

    from matbrain.mcp.artifacts import get_store

    ids, structures = [], []
    if args.handles_file:
        store = get_store()
        for h in open(args.handles_file).read().split():
            ids.append(h)
            structures.append(store.resolve_structure(h))
    else:
        for name in sorted(os.listdir(args.cif_dir)):
            if name.endswith(".cif"):
                ids.append(name)
                structures.append(chem.parse_structure(os.path.join(args.cif_dir, name)))
    funnel = ScreeningFunnel()
    ranked = funnel.run(structures, ids)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "funnel.json"), "w") as fh:
        json.dump({"stages": funnel.report(), "ranked": [{"id": c.id, **c.props} for c in ranked]}, fh, indent=2, default=str)
    for s in funnel.report():
        print(f"{s['name']:<26} {s['n_in']:>7} -> {s['n_out']:>7} ({100 * s['pass_rate']:.2f}%)  {s['seconds']:.1f}s")
    for i, c in enumerate(ranked[:10], 1):
        os.makedirs(os.path.join(args.out_dir, "cifs"), exist_ok=True)
        c.structure.to(filename=os.path.join(args.out_dir, "cifs", f"{i:02d}_{c.props['formula']}.cif"))
        print(i, c.props)


if __name__ == "__main__":
    main()
