"""PhaseDiagram tool: thermodynamic stability (energy above hull).

Combines (i) an energy for the candidate (given, or from a MatGL potential),
(ii) reference entries for the chemical system (a local frozen snapshot, or
the Materials Project API) and (iii) pymatgen's PhaseDiagram. This is the tool
used for the Li-Zr-Cl case study (Fig. 5c) and the Ehull <= 0.025 eV/atom
filter of the NRR screening funnel (Fig. 6b).

For leakage-controlled benchmarking set MATBRAIN_REFERENCE_ENTRIES to a
snapshot built from the *training split only* so that held-out materials can
never appear on the reference hull.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Literal

from pydantic import Field, model_validator
from pymatgen.analysis.phase_diagram import PDEntry, PhaseDiagram
from pymatgen.core import Composition
from pymatgen.entries.computed_entries import ComputedStructureEntry

from matbrain.mcp.artifacts import get_store
from matbrain.mcp.registry import COMPUTATION, REGISTRY, ChemSys, CifInput, Formula, ToolArgs, ToolError, require

ENERGY_MODELS = ("CHGNet", "M3GNet", "TensorNet")


# --------------------------------------------------------------------------- #
# Reference entries
# --------------------------------------------------------------------------- #
def _entry_from_record(rec: dict[str, Any]):
    if "@class" in rec:
        from monty.json import MontyDecoder

        return MontyDecoder().process_decoded(rec)
    comp = Composition(rec.get("composition") or rec.get("formula"))
    if "energy_per_atom" in rec:
        energy = float(rec["energy_per_atom"]) * comp.num_atoms
    else:
        energy = float(rec["energy"])
    return PDEntry(comp, energy, name=rec.get("name") or rec.get("material_id") or comp.reduced_formula)


@lru_cache(maxsize=4)
def load_local_reference(path: str) -> tuple:
    entries = []
    with open(path, encoding="utf-8") as fh:
        text = fh.read().strip()
    records = json.loads(text) if text.startswith("[") else [json.loads(line) for line in text.splitlines() if line.strip()]
    for rec in records:
        entries.append(_entry_from_record(rec))
    return tuple(entries)


def _subsystem(entries, elements: set[str]):
    return [e for e in entries if {el.symbol for el in e.composition.elements} <= elements]


@lru_cache(maxsize=256)
def _mp_entries(chemsys: str) -> tuple:
    mp_api = require("mp_api.client", "data")
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise ToolError("MP_API_KEY is not set and no local reference snapshot (MATBRAIN_REFERENCE_ENTRIES) is configured")
    with mp_api.MPRester(key) as mpr:
        entries = mpr.get_entries_in_chemsys(chemsys.split("-"), additional_criteria={"thermo_types": ["GGA_GGA+U"]})
    return tuple(entries)


def reference_entries(elements: set[str], source: str = "auto") -> tuple[list, str]:
    local = os.environ.get("MATBRAIN_REFERENCE_ENTRIES")
    if source in {"auto", "local"} and local:
        return _subsystem(load_local_reference(local), elements), f"local:{os.path.basename(local)}"
    if source == "local":
        raise ToolError("MATBRAIN_REFERENCE_ENTRIES is not set")
    return list(_mp_entries("-".join(sorted(elements)))), "materials_project"


def _missing_terminals(entries, elements: set[str]) -> list[str]:
    have = {e.composition.elements[0].symbol for e in entries if len(e.composition.elements) == 1}
    return sorted(elements - have)


# --------------------------------------------------------------------------- #
# Candidate energies
# --------------------------------------------------------------------------- #
def mp2020_corrected_entry(structure, energy_per_atom: float):
    """Wrap an ML (GGA-like, uncorrected) energy into an MP2020-corrected entry.

    CHGNet/M3GNet are trained on raw MPtrj GGA(+U) energies, so the MP2020
    anion/U corrections must be applied before comparing with MP entries.
    """
    from pymatgen.entries.compatibility import MaterialsProject2020Compatibility
    from pymatgen.io.vasp.sets import MPRelaxSet

    params: dict[str, Any] = {"run_type": "GGA", "hubbards": {}}
    try:
        incar = MPRelaxSet(structure).incar
        if incar.get("LDAU"):
            symbols = [str(el) for el in MPRelaxSet(structure).poscar.site_symbols]
            params = {"run_type": "GGA+U", "hubbards": {s: u for s, u in zip(symbols, incar.get("LDAUU", [])) if u}}
    except Exception:
        pass
    entry = ComputedStructureEntry(structure, energy_per_atom * len(structure), parameters=params)
    try:
        compat = MaterialsProject2020Compatibility(check_potcar=False)
        processed = compat.process_entry(entry)
        return processed if processed is not None else entry
    except TypeError:  # very old pymatgen without check_potcar
        return entry


def ehull_report(entry, ref_entries: list, source: str) -> dict[str, Any]:
    elements = {el.symbol for el in entry.composition.elements}
    missing = _missing_terminals(ref_entries, elements)
    if missing:
        raise ToolError(f"reference set lacks elemental entries for {missing}; cannot build the hull")
    pd = PhaseDiagram(list(ref_entries) + [entry])
    decomp, e_hull = pd.get_decomp_and_e_above_hull(entry, allow_negative=True)
    return {
        "formula": entry.composition.reduced_formula,
        "energy_above_hull": round(float(e_hull), 5),
        "formation_energy_per_atom": round(float(pd.get_form_energy_per_atom(entry)), 5),
        "is_stable": bool(e_hull <= 1e-8),
        "decomposition": {d.composition.reduced_formula: round(float(f), 4) for d, f in decomp.items()},
        "reference": source,
        "n_reference_entries": len(ref_entries),
        "unit": "eV/atom",
    }


class EhullArgs(ToolArgs):
    cif: CifInput | None = None
    formula: Formula | None = None
    energy_per_atom: float | None = Field(None, description="Known total energy per atom (eV); skips ML evaluation")
    energy_model: Literal["CHGNet", "M3GNet", "TensorNet"] = "CHGNet"
    relax: bool = Field(True, description="Relax with the ML potential before evaluating the energy")
    apply_mp2020: bool = Field(True, description="Apply MP2020 corrections to ML energies (needed with MP references)")
    reference: Literal["auto", "local", "mp"] = "auto"

    @model_validator(mode="after")
    def _need_input(self):
        if self.cif is None and (self.formula is None or self.energy_per_atom is None):
            raise ValueError("provide either `cif`, or both `formula` and `energy_per_atom`")
        return self


def candidate_entry(structure=None, formula=None, energy_per_atom=None, energy_model="CHGNet", relax=True, apply_mp2020=True):
    if energy_per_atom is not None and structure is None:
        comp = Composition(formula)
        return PDEntry(comp, energy_per_atom * comp.num_atoms, name=comp.reduced_formula)
    if energy_per_atom is None:
        from matbrain.mcp.tools.properties import relax_with_matgl, static_energy_per_atom

        if relax:
            structure, energy_per_atom, _ = relax_with_matgl(structure, energy_model)
        else:
            energy_per_atom = static_energy_per_atom(structure, energy_model)
        if apply_mp2020:
            return mp2020_corrected_entry(structure, energy_per_atom)
    return ComputedStructureEntry(structure, energy_per_atom * len(structure))


@REGISTRY.tool(category=COMPUTATION, timeout=1800)
def phase_diagram_ehull(args: EhullArgs) -> dict:
    """Energy above the convex hull (eV/atom), formation energy and decomposition products
    of a candidate against reference phases of its chemical system (pymatgen PhaseDiagram)."""
    structure = get_store().resolve_structure(args.cif) if args.cif else None
    entry = candidate_entry(structure, args.formula, args.energy_per_atom, args.energy_model, args.relax, args.apply_mp2020)
    elements = {el.symbol for el in entry.composition.elements}
    refs, source = reference_entries(elements, args.reference)
    return ehull_report(entry, refs, source)


class StablePhasesArgs(ToolArgs):
    chemsys: ChemSys
    reference: Literal["auto", "local", "mp"] = "auto"
    max_e_above_hull: float = Field(0.0, ge=0, le=1)


@REGISTRY.tool(category=COMPUTATION)
def phase_diagram_stable_phases(args: StablePhasesArgs) -> dict:
    """Construct the phase diagram of a chemical system (e.g. Li-Zr-Cl) and list the
    stable (and near-stable) phases with their formation energies."""
    elements = set(args.chemsys.split("-"))
    refs, source = reference_entries(elements, args.reference)
    missing = _missing_terminals(refs, elements)
    if missing:
        raise ToolError(f"reference set lacks elemental entries for {missing}")
    pd = PhaseDiagram(refs)
    rows = []
    for e in refs:
        eh = pd.get_e_above_hull(e, allow_negative=True)
        if eh is not None and eh <= args.max_e_above_hull + 1e-8:
            rows.append({"formula": e.composition.reduced_formula, "e_above_hull": round(float(eh), 4), "formation_energy_per_atom": round(float(pd.get_form_energy_per_atom(e)), 4)})
    uniq = {r["formula"]: r for r in sorted(rows, key=lambda r: r["e_above_hull"], reverse=True)}
    return {"chemsys": args.chemsys, "reference": source, "phases": sorted(uniq.values(), key=lambda r: r["formation_energy_per_atom"])}


def ehull_batch(entries: list, source: str = "auto") -> list[dict[str, Any]]:
    """Energy above hull for many candidates, building one hull per chemical system.

    Candidates are not added to the hull, so a candidate below the reference
    hull gets a negative value (allow_negative=True).
    """
    by_sys: dict[frozenset, list[int]] = {}
    for i, e in enumerate(entries):
        by_sys.setdefault(frozenset(el.symbol for el in e.composition.elements), []).append(i)
    out: list[dict[str, Any]] = [{} for _ in entries]
    for elements, idx in by_sys.items():
        refs, src = reference_entries(set(elements), source)
        missing = _missing_terminals(refs, set(elements))
        if missing:
            for i in idx:
                out[i] = {"energy_above_hull": None, "reference": src, "error": f"missing elemental references {missing}"}
            continue
        pd = PhaseDiagram(refs)
        for i in idx:
            _, eh = pd.get_decomp_and_e_above_hull(entries[i], allow_negative=True)
            out[i] = {"energy_above_hull": float(eh), "reference": src}
    return out
