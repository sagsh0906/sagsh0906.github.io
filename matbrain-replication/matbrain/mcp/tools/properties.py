"""Property-prediction and simulation tools.

Backends (all optional, imported lazily inside the Mat-MCP container):

* CHGNet (standalone ``chgnet`` package, v0.3.0 weights bundled with the
  wheel): the default universal potential. It is trained on MPtrj
  (Materials Project GGA/GGA+U), so its energies are MP-compatible after the
  MP2020 corrections - which the E_hull tool relies on.
* MatGL (>= 3.0 downloads all weights from the ``materialyze`` Hugging Face
  org): M3GNet / TensorNet potentials (MatPES-trained, i.e. *not* MP2020
  compatible), MEGNet / M3GNet formation-energy and MEGNet band-gap models.
  Model names changed across MatGL releases, so each model is resolved from a
  list of aliases; override with MATBRAIN_<KIND>_<FAMILY>_MODEL, e.g.
  MATBRAIN_BANDGAP_MEGNET_MODEL (see ``matgl.get_available_pretrained_models()``).
* MatterSim, FairChem, and VASP input generation through pymatgen.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic import Field

from matbrain.mcp.artifacts import get_store
from matbrain.mcp.registry import COMPUTATION, REGISTRY, CifInput, ToolArgs, ToolError, require

# New (MatGL >= 3) names first, legacy names after.
PES_ALIASES = {
    "M3GNet": ["M3GNet-PES-MatPES-PBE-2025.2", "M3GNet-MatPES-PBE-v2025.1-PES", "M3GNet-MP-2021.2.8-PES"],
    "TensorNet": ["TensorNet-PES-MatPES-PBE-2025.2", "TensorNet-MatPES-PBE-v2025.1-PES"],
}
EFORM_ALIASES = {
    "MEGNet": ["MEGNet-Eform-MP-2018.6.1", "MEGNet-MP-2018.6.1-Eform"],
    "M3GNet": ["M3GNet-Eform-MP-2018.6.1", "M3GNet-MP-2018.6.1-Eform"],
}
BANDGAP_ALIASES = ["MEGNet-BandGap-mfi-MP-2019.4.1", "MEGNet-MP-2019.4.1-BandGap-mfi"]
BANDGAP_FIDELITY = {"PBE": 0, "GLLB-SC": 1, "HSE": 2, "SCAN": 3}
PES_MODELS = Literal["CHGNet", "M3GNet", "TensorNet"]


def _device() -> str:
    return os.environ.get("MATBRAIN_DEVICE", "cuda" if _cuda_available() else "cpu")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@lru_cache(maxsize=16)
def load_matgl_model(kind: str, family: str):
    matgl = require("matgl")
    env = os.environ.get(f"MATBRAIN_{kind.upper()}_{family.upper()}_MODEL")
    table = {"pes": PES_ALIASES, "eform": EFORM_ALIASES}.get(kind)
    aliases = [env] if env else (BANDGAP_ALIASES if kind == "bandgap" else table.get(family, []))
    if not aliases:
        raise ToolError(f"no MatGL {kind} model for family {family}")
    errors = []
    for name in aliases:
        try:
            return matgl.load_model(name)
        except Exception as exc:  # wrong name for this matgl version, or download failure
            errors.append(f"{name}: {exc}")
    raise ToolError(f"could not load MatGL model (set MATBRAIN_{kind.upper()}_{family.upper()}_MODEL); tried " + " | ".join(errors))


@lru_cache(maxsize=1)
def _chgnet():
    chgnet_model = require("chgnet.model")
    return chgnet_model.CHGNet.load(use_device=_device(), verbose=False)


def relax_with_potential(structure, model: str = "CHGNet", fmax: float = 0.1, steps: int = 500, relax_cell: bool = True):
    """Returns (final structure, energy per atom in eV, converged)."""
    if model == "CHGNet":
        from chgnet.model import StructOptimizer

        res = StructOptimizer(model=_chgnet(), use_device=_device()).relax(structure, fmax=fmax, steps=steps, relax_cell=relax_cell, verbose=False)
    else:
        require("matgl")
        from matgl.ext.ase import Relaxer

        res = Relaxer(potential=load_matgl_model("pes", model), relax_cell=relax_cell).relax(structure, fmax=fmax, steps=steps)
    final, traj = res["final_structure"], res["trajectory"]
    return final, float(traj.energies[-1]) / len(final), len(traj.energies) < steps


def static_energy_per_atom(structure, model: str = "CHGNet") -> float:
    if model == "CHGNet":
        return float(_chgnet().predict_structure(structure)["e"])
    require("matgl")
    from matgl.ext.ase import PESCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = PESCalculator(load_matgl_model("pes", model))
    return float(atoms.get_potential_energy()) / len(atoms)


# Backwards-compatible name used by earlier callers.
relax_with_matgl = relax_with_potential


def _struct(cif: str):
    return get_store().resolve_structure(cif)


# --------------------------------------------------------------------------- #
class RelaxArgs(ToolArgs):
    cif: CifInput
    model: PES_MODELS = "CHGNet"
    fmax: float = Field(0.1, gt=0, le=1, description="Force convergence criterion (eV/A)")
    steps: int = Field(500, ge=1, le=5000)
    relax_cell: bool = True


@REGISTRY.tool(category=COMPUTATION, optional_deps=("chgnet", "matgl"), timeout=1800)
def relax_structure(args: RelaxArgs) -> dict:
    """Relax a structure with a universal potential (CHGNet by default; M3GNet / TensorNet via MatGL).
    Returns a handle to the relaxed structure, energy per atom and convergence flag."""
    s = _struct(args.cif)
    if not s.is_ordered:
        raise ToolError("relaxation requires an ordered structure (no fractional occupancies)")
    final, e_pa, converged = relax_with_potential(s, args.model, args.fmax, args.steps, args.relax_cell)
    handle = get_store().put_structure(final, {"source": f"relax:{args.model}", "energy_per_atom": e_pa})
    return {
        "handle": handle,
        "energy_per_atom": round(e_pa, 5),
        "converged": converged,
        "volume_change": round(final.volume / s.volume - 1, 4),
        "formula": final.composition.reduced_formula,
    }


class EnergyArgs(ToolArgs):
    cif: CifInput
    model: PES_MODELS = "CHGNet"


@REGISTRY.tool(category=COMPUTATION, optional_deps=("chgnet", "matgl"))
def predict_energy(args: EnergyArgs) -> dict:
    """Static (single-point) total energy per atom from a universal potential."""
    s = _struct(args.cif)
    return {"energy_per_atom": round(static_energy_per_atom(s, args.model), 5), "model": args.model}


class EformArgs(ToolArgs):
    cif: CifInput
    model: Literal["MEGNet", "M3GNet"] = "MEGNet"


@REGISTRY.tool(category=COMPUTATION, optional_deps=("matgl",))
def predict_formation_energy(args: EformArgs) -> dict:
    """Formation energy per atom (eV/atom) predicted by a MatGL property model trained on MP."""
    s = _struct(args.cif)
    model = load_matgl_model("eform", args.model)
    value = float(model.predict_structure(s))
    return {"formation_energy_per_atom": round(value, 4), "unit": "eV/atom", "model": args.model}


class BandgapArgs(ToolArgs):
    cif: CifInput
    fidelity: Literal["PBE", "GLLB-SC", "HSE", "SCAN"] = "PBE"


@REGISTRY.tool(category=COMPUTATION, optional_deps=("matgl", "torch"))
def predict_bandgap(args: BandgapArgs) -> dict:
    """Band gap (eV) from the multi-fidelity MEGNet model; also reports metallicity (gap < 0.05 eV)."""
    torch = require("torch", "analysis")
    s = _struct(args.cif)
    model = load_matgl_model("bandgap", "MEGNet")
    gap = float(model.predict_structure(structure=s, state_attr=torch.tensor([BANDGAP_FIDELITY[args.fidelity]])))
    gap = max(gap, 0.0)
    return {"band_gap": round(gap, 4), "unit": "eV", "fidelity": args.fidelity, "is_metal": gap < 0.05}


class MagmomArgs(ToolArgs):
    cif: CifInput
    threshold: float = Field(0.5, ge=0, description="|m| (mu_B) above which a site is counted as magnetic")


@REGISTRY.tool(category=COMPUTATION, optional_deps=("chgnet",))
def predict_magnetic_moments(args: MagmomArgs) -> dict:
    """Site magnetic-moment magnitudes predicted by CHGNet and a magnetic/non-magnetic call.
    CHGNet predicts |m| only, so collinear ordering (FM/AFM/FiM) is not determined."""
    s = _struct(args.cif)
    pred = _chgnet().predict_structure(s)
    mags = [float(abs(m)) for m in pred["m"]]
    n_mag = sum(m > args.threshold for m in mags)
    return {
        "site_magmoms": [round(m, 3) for m in mags],
        "n_magnetic_sites": n_mag,
        "total_abs_magnetization_per_atom": round(sum(mags) / len(mags), 4),
        "is_magnetic": n_mag > 0,
        "energy_per_atom": round(float(pred["e"]), 5),
    }


class MatterSimArgs(ToolArgs):
    cif: CifInput
    relax: bool = False


@REGISTRY.tool(category=COMPUTATION, optional_deps=("mattersim",), timeout=1800)
def mattersim_predict(args: MatterSimArgs) -> dict:
    """Energy (and optional relaxation) with the MatterSim universal potential."""
    require("mattersim")
    from mattersim.forcefield import MatterSimCalculator
    from pymatgen.io.ase import AseAtomsAdaptor

    s = _struct(args.cif)
    atoms = AseAtomsAdaptor.get_atoms(s)
    atoms.calc = MatterSimCalculator(device=_device())
    if args.relax:
        from ase.filters import FrechetCellFilter
        from ase.optimize import FIRE

        FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.05, steps=500)
    out = {"energy_per_atom": round(float(atoms.get_potential_energy()) / len(atoms), 5)}
    if args.relax:
        out["handle"] = get_store().put_structure(AseAtomsAdaptor.get_structure(atoms), {"source": "mattersim_relax"})
    return out


class FairChemArgs(ToolArgs):
    cif: CifInput
    model_name: str = Field("uma-s-1", description="FAIRChem pretrained model name")
    task_name: str = Field("omat", description="UMA task head (omat for inorganic materials)")
    fmax: float = Field(0.05, gt=0)
    steps: int = Field(300, ge=1, le=5000)


@REGISTRY.tool(category=COMPUTATION, optional_deps=("fairchem",), timeout=1800)
def fairchem_optimize(args: FairChemArgs) -> dict:
    """Structural optimisation with a FAIRChem (UMA) potential."""
    require("fairchem.core")
    from ase.filters import FrechetCellFilter
    from ase.optimize import FIRE
    from pymatgen.io.ase import AseAtomsAdaptor

    try:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        unit = pretrained_mlip.get_predict_unit(args.model_name, device=_device())
        calc = FAIRChemCalculator(unit, task_name=args.task_name)
    except ImportError as exc:  # fairchem-core v1 API
        raise ToolError(f"fairchem-core>=2 is required: {exc}") from exc
    atoms = AseAtomsAdaptor.get_atoms(_struct(args.cif))
    atoms.calc = calc
    FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=args.fmax, steps=args.steps)
    s = AseAtomsAdaptor.get_structure(atoms)
    return {
        "handle": get_store().put_structure(s, {"source": "fairchem"}),
        "energy_per_atom": round(float(atoms.get_potential_energy()) / len(atoms), 5),
    }


class VaspArgs(ToolArgs):
    cif: CifInput
    calc_type: Literal["relax", "static", "band_structure"] = "relax"
    output_dir: str = Field("vasp_runs", description="Directory (inside the tool container) to write inputs to")


@REGISTRY.tool(category=COMPUTATION)
def vasp_prepare_inputs(args: VaspArgs) -> dict:
    """Write MP-compatible VASP inputs (INCAR/KPOINTS/POSCAR[/POTCAR]) for a
    high-fidelity DFT calculation. Job submission is site-specific and left to the user."""
    from pymatgen.io.vasp.sets import MPNonSCFSet, MPRelaxSet, MPStaticSet

    s = _struct(args.cif)
    cls = {"relax": MPRelaxSet, "static": MPStaticSet, "band_structure": MPNonSCFSet}[args.calc_type]
    stem = get_store().put_structure(s).removeprefix("cif://")
    out = os.path.join(args.output_dir, f"{stem}_{args.calc_type}")
    has_psp = bool(os.environ.get("PMG_VASP_PSP_DIR"))
    kwargs = {"mode": "line"} if args.calc_type == "band_structure" else {}
    vis = cls(s, **kwargs)
    vis.write_input(out, potcar_spec=not has_psp)
    return {"directory": out, "files": sorted(os.listdir(out)), "potcar_written": has_psp}


class CustomModelArgs(ToolArgs):
    cif: CifInput
    property: Literal["efermi", "custom"] = Field("efermi", description="Which configured model to use")


@REGISTRY.tool(category=COMPUTATION, optional_deps=("matgl",))
def predict_property_custom(args: CustomModelArgs) -> dict:
    """Predict a scalar property with a locally trained MatGL model (e.g. a MEGNet/M3GNet
    model fitted to MP Fermi energies). Configure MATBRAIN_EFERMI_MODEL / MATBRAIN_CUSTOM_MODEL."""
    matgl = require("matgl")
    path = os.environ.get("MATBRAIN_EFERMI_MODEL" if args.property == "efermi" else "MATBRAIN_CUSTOM_MODEL")
    if not path:
        raise ToolError(f"no model configured for property '{args.property}'")
    model = _load_path_model(path, matgl)
    return {"property": args.property, "value": round(float(model.predict_structure(_struct(args.cif))), 4)}


@lru_cache(maxsize=4)
def _load_path_model(path: str, matgl):
    return matgl.load_model(path)
