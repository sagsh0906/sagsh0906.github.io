"""Data-acquisition tools (Materials Project, OQMD).

These are *target-database lookup* tools: they are disabled in the
leakage-controlled benchmark (``ToolRegistry.subset(disable_target_db=True)``)
so that no system can read held-out labels.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx
from pydantic import Field, model_validator
from pymatgen.core import Composition, Lattice, Structure

from matbrain.mcp.artifacts import get_store
from matbrain.mcp.registry import REGISTRY, RETRIEVAL, ChemSys, Formula, ToolArgs, ToolError, require

MP_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "symmetry",
    "nsites",
    "formation_energy_per_atom",
    "energy_above_hull",
    "band_gap",
    "is_metal",
    "is_magnetic",
    "ordering",
    "efermi",
    "theoretical",
]


def _mp_client():
    mp_api = require("mp_api.client", "data")
    key = os.environ.get("MP_API_KEY")
    if not key:
        raise ToolError("MP_API_KEY is not configured in the Mat-MCP environment")
    return mp_api.MPRester(key)


def _summarize_mp_doc(doc: Any) -> dict[str, Any]:
    d = doc if isinstance(doc, dict) else doc.model_dump()
    sym = d.get("symmetry") or {}
    out = {
        "material_id": str(d.get("material_id")),
        "formula": d.get("formula_pretty"),
        "space_group": sym.get("symbol"),
        "space_group_number": sym.get("number"),
        "crystal_system": str(sym.get("crystal_system")) if sym.get("crystal_system") else None,
        "nsites": d.get("nsites"),
        "formation_energy_per_atom": d.get("formation_energy_per_atom"),
        "energy_above_hull": d.get("energy_above_hull"),
        "band_gap": d.get("band_gap"),
        "is_metal": d.get("is_metal"),
        "is_magnetic": d.get("is_magnetic"),
        "ordering": str(d.get("ordering")) if d.get("ordering") is not None else None,
        "efermi": d.get("efermi"),
        "theoretical": d.get("theoretical"),
    }
    structure = d.get("structure")
    if structure is not None:
        s = structure if isinstance(structure, Structure) else Structure.from_dict(structure)
        out["handle"] = get_store().put_structure(s, {"source": "materials_project", "material_id": out["material_id"]})
    return out


class MPSearchArgs(ToolArgs):
    formula: Formula | None = None
    chemsys: ChemSys | None = None
    material_id: str | None = Field(None, pattern=r"^mp-\d+$")
    spacegroup_number: int | None = Field(None, ge=1, le=230)
    max_e_above_hull: float | None = Field(None, ge=0)
    max_results: int = Field(5, ge=1, le=100)

    @model_validator(mode="after")
    def _one_query(self):
        if not (self.formula or self.chemsys or self.material_id):
            raise ValueError("provide formula, chemsys or material_id")
        return self


@REGISTRY.tool(category=RETRIEVAL, target_db_lookup=True, optional_deps=("mp_api",))
def mp_search(args: MPSearchArgs) -> dict:
    """Search the Materials Project by formula, chemical system or material id. Returns
    computed properties and artifact handles for the structures."""
    comp = Composition(args.formula) if args.formula else None
    if comp is not None and any(abs(v - round(v)) > 1e-6 for v in comp.values()):
        raise ToolError("Non-stoichiometric compositions are not supported by the Materials Project formula search")
    query: dict[str, Any] = {"fields": MP_FIELDS}
    if args.material_id:
        query["material_ids"] = [args.material_id]
    if args.formula:
        query["formula"] = comp.reduced_formula
    if args.chemsys:
        query["chemsys"] = args.chemsys
    if args.spacegroup_number:
        query["spacegroup_number"] = args.spacegroup_number
    if args.max_e_above_hull is not None:
        query["energy_above_hull"] = (0, args.max_e_above_hull)
    with _mp_client() as mpr:
        docs = mpr.materials.summary.search(**query)
    docs = sorted(docs, key=lambda d: (getattr(d, "energy_above_hull", None) or 0.0))[: args.max_results]
    return {"n_results": len(docs), "results": [_summarize_mp_doc(d) for d in docs]}


# --------------------------------------------------------------------------- #
OQMD_URL = os.environ.get("OQMD_URL", "https://oqmd.org/oqmdapi/formationenergy")
_SITE_RE = re.compile(r"^\s*([A-Z][a-z]?)\s*@\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)")


def oqmd_structure(rec: dict[str, Any]) -> Structure | None:
    cell, sites = rec.get("unit_cell"), rec.get("sites")
    if not cell or not sites:
        return None
    species, coords = [], []
    for site in sites:
        m = _SITE_RE.match(site)
        if not m:
            return None
        species.append(m.group(1))
        coords.append([float(m.group(i)) for i in (2, 3, 4)])
    return Structure(Lattice(cell), species, coords)


class OQMDSearchArgs(ToolArgs):
    formula: Formula
    max_stability: float | None = Field(None, description="Upper bound on OQMD hull distance (eV/atom)")
    max_results: int = Field(5, ge=1, le=100)


@REGISTRY.tool(category=RETRIEVAL, target_db_lookup=True)
def oqmd_search(args: OQMDSearchArgs) -> dict:
    """Search the Open Quantum Materials Database (OQMD) REST API by composition."""
    comp = Composition(args.formula)
    if any(abs(v - round(v)) > 1e-6 for v in comp.values()):
        raise ToolError("Non-stoichiometric compositions are not supported by the OQMD composition filter")
    params = {
        "composition": comp.reduced_formula,
        "fields": "name,entry_id,spacegroup,delta_e,stability,band_gap,natoms,unit_cell,sites",
        "limit": args.max_results,
        "format": "json",
    }
    if args.max_stability is not None:
        params["filter"] = f"stability<={args.max_stability}"
    resp = httpx.get(OQMD_URL, params=params, timeout=60)
    resp.raise_for_status()
    rows = []
    for rec in resp.json().get("data", []):
        row = {k: rec.get(k) for k in ("name", "entry_id", "spacegroup", "delta_e", "stability", "band_gap", "natoms")}
        s = oqmd_structure(rec)
        if s is not None:
            row["handle"] = get_store().put_structure(s, {"source": "oqmd", "entry_id": rec.get("entry_id")})
        rows.append(row)
    return {"n_results": len(rows), "results": rows}
