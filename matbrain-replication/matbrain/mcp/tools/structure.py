"""Structural validation and analysis tools (pymatgen based, CPU only)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition

from matbrain import chem
from matbrain.mcp.artifacts import get_store
from matbrain.mcp.registry import ANALYSIS, REGISTRY, VALIDATION, CifInput, Formula, ToolArgs, ToolError


def _structure(cif_or_handle: str):
    return get_store().resolve_structure(cif_or_handle)


# --------------------------------------------------------------------------- #
class ValidateStructureArgs(ToolArgs):
    cif: CifInput
    min_distance: float = Field(chem.MIN_BOND_DISTANCE, gt=0, le=3, description="Minimum allowed interatomic distance (A)")
    symprec: float = Field(0.1, gt=0, le=1)


@REGISTRY.tool(category=VALIDATION)
def validate_structure(args: ValidateStructureArgs) -> dict:
    """Validate a crystal structure: CIF parseability, interatomic overlaps, volume,
    oxidation-state (charge) balance and symmetry. Returns a JSON report."""
    s = _structure(args.cif)
    report = {"formula": s.composition.reduced_formula, "full_formula": s.composition.formula.replace(" ", "")}
    report.update(chem.structural_validity(s, args.min_distance))
    report.update(chem.charge_balance(s.composition))
    try:
        report.update(chem.symmetry_info(s, symprec=args.symprec))
    except Exception as exc:  # spglib can fail on pathological cells
        report["symmetry_error"] = str(exc)
    report["is_ordered"] = s.is_ordered
    return report


class SymmetryArgs(ToolArgs):
    cif: CifInput
    symprec: float = Field(0.1, gt=0, le=1)
    angle_tolerance: float = Field(5.0, gt=0, le=20)


@REGISTRY.tool(category=ANALYSIS)
def analyze_symmetry(args: SymmetryArgs) -> dict:
    """Space group, crystal system, lattice type, point group, Wyckoff-based
    protostructure label and lattice parameters of a structure."""
    s = _structure(args.cif)
    info = chem.symmetry_info(s, args.symprec, args.angle_tolerance)
    try:
        info["protostructure"] = chem.protostructure_label(s, symprec=args.symprec)
    except Exception as exc:
        info["protostructure_error"] = str(exc)
    a, b, c = s.lattice.abc
    al, be, ga = s.lattice.angles
    info["lattice"] = {"a": round(a, 5), "b": round(b, 5), "c": round(c, 5), "alpha": round(al, 3), "beta": round(be, 3), "gamma": round(ga, 3), "volume": round(s.volume, 4)}
    info["num_sites"] = len(s)
    return info


class ChargeBalanceArgs(ToolArgs):
    formula: Formula


@REGISTRY.tool(category=VALIDATION)
def check_charge_balance(args: ChargeBalanceArgs) -> dict:
    """Check whether a composition admits a charge-neutral assignment of common
    oxidation states (valence plausibility). Fractional formulas are scaled to integers."""
    return {"formula": args.formula, **chem.charge_balance(args.formula)}


class MatchArgs(ToolArgs):
    cif_a: CifInput
    cif_b: CifInput
    ltol: float = Field(0.2, gt=0)
    stol: float = Field(0.3, gt=0)
    angle_tol: float = Field(5.0, gt=0)


@REGISTRY.tool(category=ANALYSIS)
def match_structures(args: MatchArgs) -> dict:
    """Test whether two structures are equivalent with pymatgen StructureMatcher."""
    sm = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)
    a, b = _structure(args.cif_a), _structure(args.cif_b)
    match = sm.fit(a, b)
    rms = sm.get_rms_dist(a, b) if match else None
    return {"match": bool(match), "rms_dist": None if rms is None else [float(x) for x in rms]}


class DedupArgs(ToolArgs):
    cifs: list[CifInput] = Field(..., min_length=1, max_length=50000)
    ltol: float = Field(0.2, gt=0)
    stol: float = Field(0.3, gt=0)
    angle_tol: float = Field(5.0, gt=0)


@REGISTRY.tool(category=ANALYSIS)
def deduplicate_structures(args: DedupArgs) -> dict:
    """Remove duplicate structures (StructureMatcher grouping). Paper defaults for
    the NRR screening: site tolerance 0.3, lattice tolerance 0.2, angle tolerance 5 deg."""
    store = get_store()
    structs = [store.resolve_structure(c) for c in args.cifs]
    sm = StructureMatcher(ltol=args.ltol, stol=args.stol, angle_tol=args.angle_tol)
    groups = sm.group_structures(structs)
    index = {id(s): i for i, s in enumerate(structs)}
    keep = sorted(index[id(g[0])] for g in groups)
    handles = [args.cifs[i] if args.cifs[i].startswith("cif://") else store.put_structure(structs[i]) for i in keep]
    return {"n_input": len(structs), "n_unique": len(keep), "unique": handles}


class GetStructureArgs(ToolArgs):
    handle: CifInput
    symmetrized: bool = Field(False, description="Write a symmetrized CIF (symprec 0.1)")


@REGISTRY.tool(category=ANALYSIS)
def get_structure_cif(args: GetStructureArgs) -> dict:
    """Return the full CIF text for an artifact handle (use for the final answer)."""
    s = _structure(args.handle)
    return {"cif": chem.structure_to_cif(s, symprec=0.1 if args.symmetrized else None)}


class VegardArgs(ToolArgs):
    end_member_values: dict[str, float] = Field(..., description="End member -> lattice parameter or volume, e.g. {'CsPbCl3': 5.67984}")
    fractions: dict[str, float] = Field(..., description="End member -> mole fraction (must sum to 1)")


@REGISTRY.tool(category=ANALYSIS)
def vegard_interpolate(args: VegardArgs) -> dict:
    """Vegard's-law linear interpolation of a lattice parameter (or volume) of a solid solution."""
    try:
        value = chem.vegard_lattice(args.end_member_values, args.fractions)
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {"value": round(value, 6), "terms": {k: round(args.end_member_values[k] * f, 6) for k, f in args.fractions.items()}}


class DisorderedPerovskiteArgs(ToolArgs):
    a_site: str = Field(..., description="A-site element, e.g. Cs")
    b_site: str = Field(..., description="B-site element, e.g. Pb")
    x_site_occupancy: dict[str, float] = Field(..., description="X-site element -> fractional occupancy, e.g. {'Cl':0.2,'Br':0.4,'I':0.4}")
    lattice_a: float = Field(..., gt=2, lt=20, description="Cubic lattice parameter in Angstrom (e.g. from vegard_interpolate)")


@REGISTRY.tool(category=ANALYSIS)
def build_disordered_perovskite(args: DisorderedPerovskiteArgs) -> dict:
    """Build a cubic Pm-3m ABX3 virtual-crystal (fractional occupancy) structure."""
    total = sum(args.x_site_occupancy.values())
    if abs(total - 1) > 1e-4:
        raise ToolError(f"X-site occupancies must sum to 1 (got {total})")
    s = chem.cubic_perovskite(args.a_site, args.b_site, args.x_site_occupancy, args.lattice_a)
    handle = get_store().put_structure(s, {"source": "vca_perovskite"})
    return {"handle": handle, "formula": s.composition.formula.replace(" ", ""), **chem.structural_validity(s)}


class PerovskiteFactorArgs(ToolArgs):
    a_site: dict[str, float]
    b_site: dict[str, float]
    x_site: dict[str, float]
    a_oxidation: int = 1
    b_oxidation: int = 2
    x_oxidation: int = -1


@REGISTRY.tool(category=ANALYSIS)
def perovskite_factors(args: PerovskiteFactorArgs) -> dict:
    """Goldschmidt tolerance factor and octahedral factor from Shannon radii
    (A: XII, B: VI, X: VI coordination; occupancy-weighted for mixed sites)."""
    try:
        res = chem.perovskite_factors(args.a_site, args.b_site, args.x_site, args.a_oxidation, args.b_oxidation, args.x_oxidation)
    except Exception as exc:
        raise ToolError(f"Shannon radius lookup failed: {exc}") from exc
    return {k: round(v, 4) for k, v in res.items()}


class SubstituteArgs(ToolArgs):
    cif: CifInput
    mapping: dict[str, str] = Field(..., description="Element substitutions, e.g. {'Na': 'K'}")
    scale_volume: bool = Field(True, description="Rescale the volume using atomic radii")


@REGISTRY.tool(category=ANALYSIS)
def substitute_species(args: SubstituteArgs) -> dict:
    """Prototype-based design: substitute elements in a template structure."""
    s = _structure(args.cif).copy()
    try:
        s.replace_species(args.mapping)
    except Exception as exc:
        raise ToolError(f"substitution failed: {exc}") from exc
    if args.scale_volume:
        from pymatgen.analysis.structure_prediction.volume_predictor import DLSVolumePredictor

        try:
            s = DLSVolumePredictor().get_predicted_structure(s)
        except Exception:
            pass
    handle = get_store().put_structure(s, {"source": "substitution"})
    return {"handle": handle, "formula": s.composition.reduced_formula, **chem.structural_validity(s)}


class SupercellArgs(ToolArgs):
    cif: CifInput
    scaling: list[int] = Field(..., min_length=3, max_length=3)


@REGISTRY.tool(category=ANALYSIS)
def make_supercell(args: SupercellArgs) -> dict:
    """Build a diagonal supercell."""
    s = _structure(args.cif) * tuple(args.scaling)
    return {"handle": get_store().put_structure(s, {"source": "supercell"}), "num_sites": len(s)}


class CompositionArgs(ToolArgs):
    formula: Formula
    mode: Literal["integerize", "reduce"] = "integerize"


@REGISTRY.tool(category=ANALYSIS)
def normalize_composition(args: CompositionArgs) -> dict:
    """Convert a (possibly fractional) formula to an integer approximant or reduced formula,
    e.g. CsPb(Cl0.2Br0.4I0.4)3 -> Cs5Pb5Cl3Br6I6."""
    comp = Composition(args.formula)
    if args.mode == "integerize":
        out = chem.integerize_composition(comp)
    else:
        out = comp.reduced_composition
    return {"input": args.formula, "output": out.formula.replace(" ", ""), "num_atoms": out.num_atoms}


class StoreStructureArgs(ToolArgs):
    cif: CifInput


@REGISTRY.tool(category=ANALYSIS)
def store_structure(args: StoreStructureArgs) -> dict:
    """Register a CIF as an artifact and return its handle (cif://...) for use by other tools."""
    s = _structure(args.cif)
    return {"handle": get_store().put_structure(s, {"source": "user"}), "formula": s.composition.reduced_formula, "num_sites": len(s)}
