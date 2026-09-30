"""Shared crystallographic utilities used by Mat-MCP tools, the leakage audit,
the benchmark metrics and the screening funnel.

Everything here depends only on pymatgen/spglib so it runs on CPU.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import warnings
from fractions import Fraction
from functools import reduce
from typing import Any

import numpy as np

# Heavy import, but every consumer of this module needs it.
from pymatgen.core import Composition, Element, Lattice, Species, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# CDVAE / DiffCSP convention for a "valid" structure.
MIN_BOND_DISTANCE = 0.5  # Angstrom
MIN_VOLUME = 0.1  # Angstrom^3

CRYSTAL_SYSTEM_RANGES = (
    (2, "triclinic"),
    (15, "monoclinic"),
    (74, "orthorhombic"),
    (142, "tetragonal"),
    (167, "trigonal"),
    (194, "hexagonal"),
    (230, "cubic"),
)

_PEARSON_FAMILY = {
    "triclinic": "a",
    "monoclinic": "m",
    "orthorhombic": "o",
    "tetragonal": "t",
    "trigonal": "h",
    "hexagonal": "h",
    "cubic": "c",
}


class StructureParseError(ValueError):
    """Raised when a CIF / structure string cannot be parsed by pymatgen."""


# --------------------------------------------------------------------------- #
# Parsing
# --------------------------------------------------------------------------- #
def looks_like_cif(text: str) -> bool:
    return bool(re.search(r"^\s*data_", text, re.M)) and "_cell_length_a" in text


def parse_structure(obj: Any) -> Structure:
    """Parse a pymatgen Structure from a Structure, dict, CIF text or file path."""
    if isinstance(obj, Structure):
        return obj
    if isinstance(obj, dict):
        return Structure.from_dict(obj)
    if not isinstance(obj, str) or not obj.strip():
        raise StructureParseError("empty structure input")
    text = obj
    if len(text) < 4096 and "\n" not in text and os.path.isfile(text):
        with open(text, encoding="utf-8") as fh:
            text = fh.read()
    fmt = "cif" if looks_like_cif(text) else None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if fmt:
                s = Structure.from_str(text, fmt="cif")
            else:
                # POSCAR or JSON
                stripped = text.lstrip()
                s = Structure.from_str(text, fmt="json" if stripped.startswith("{") else "poscar")
    except Exception as exc:  # pymatgen raises many different exception types
        raise StructureParseError(f"pymatgen could not parse structure: {exc}") from exc
    if len(s) == 0:
        raise StructureParseError("structure has no sites")
    return s


def structure_to_cif(structure: Structure, symprec: float | None = None) -> str:
    from pymatgen.io.cif import CifWriter

    return str(CifWriter(structure, symprec=symprec))


def normalize_cif_text(cif: str) -> str:
    """Whitespace/comment-insensitive normalisation used for exact CIF hashing."""
    lines = []
    for line in cif.splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            lines.append(" ".join(line.split()))
    return "\n".join(lines)


def cif_hash(cif: str) -> str:
    """SHA-256 of the normalised CIF text (audit level 2: exact CIF-hash overlap)."""
    return hashlib.sha256(normalize_cif_text(cif).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Geometry / validity
# --------------------------------------------------------------------------- #
def min_interatomic_distance(structure: Structure) -> float:
    """Minimum distance between any two sites, including periodic self images."""
    self_image = min(structure.lattice.get_niggli_reduced_lattice().abc)
    if len(structure) == 1:
        return float(self_image)
    dm = np.array(structure.distance_matrix, dtype=float)
    np.fill_diagonal(dm, np.inf)
    return float(min(dm.min(), self_image))


def structural_validity(structure: Structure, min_dist: float = MIN_BOND_DISTANCE) -> dict[str, Any]:
    """Geometric validity check (paper: 'excluding physically impossible atomic overlaps')."""
    dmin = min_interatomic_distance(structure)
    vol = float(structure.volume)
    density = float(structure.density)
    ok = dmin >= min_dist and vol >= MIN_VOLUME and math.isfinite(density)
    return {
        "valid": bool(ok),
        "min_distance": round(dmin, 4),
        "volume": round(vol, 4),
        "density": round(density, 4),
        "num_sites": len(structure),
    }


# --------------------------------------------------------------------------- #
# Composition / charge balance
# --------------------------------------------------------------------------- #
def integerize_composition(comp: Composition, max_denominator: int = 100) -> Composition:
    """Scale a fractional composition (e.g. CsPbCl0.6Br1.2I1.2) to integers."""
    fracs = {el: Fraction(amt).limit_denominator(max_denominator) for el, amt in comp.items()}
    lcm = reduce(lambda a, b: a * b // math.gcd(a, b), (f.denominator for f in fracs.values()), 1)
    return Composition({el: int(f * lcm) for el, f in fracs.items()})


def charge_balance(formula: str | Composition, max_reduced_atoms: int = 60) -> dict[str, Any]:
    """Whether a composition admits a charge-neutral assignment of common oxidation states."""
    comp = formula if isinstance(formula, Composition) else Composition(formula)
    comp = comp.element_composition
    if any(abs(a - round(a)) > 1e-6 for a in comp.values()):
        comp = integerize_composition(comp)
    reduced = comp.reduced_composition
    if reduced.num_atoms > max_reduced_atoms:
        return {"charge_balanced": False, "oxidation_states": None, "reason": "composition too large to enumerate"}
    if len(reduced.elements) == 1:
        # Elemental solids are trivially neutral.
        return {"charge_balanced": True, "oxidation_states": {str(reduced.elements[0]): 0}, "reason": "elemental"}
    try:
        guesses = reduced.oxi_state_guesses(max_sites=-1)
    except Exception as exc:
        return {"charge_balanced": False, "oxidation_states": None, "reason": f"oxi_state_guesses failed: {exc}"}
    if not guesses:
        return {"charge_balanced": False, "oxidation_states": None, "reason": "no neutral assignment"}
    return {"charge_balanced": True, "oxidation_states": dict(guesses[0]), "reason": "ok"}


def reduced_formula(formula_or_structure: Any) -> str:
    if isinstance(formula_or_structure, Structure):
        return formula_or_structure.composition.reduced_formula
    return Composition(str(formula_or_structure)).reduced_formula


def chemical_system(formula_or_structure: Any) -> str:
    comp = (
        formula_or_structure.composition
        if isinstance(formula_or_structure, Structure)
        else Composition(str(formula_or_structure))
    )
    return "-".join(sorted(el.symbol for el in comp.elements))


# --------------------------------------------------------------------------- #
# Symmetry
# --------------------------------------------------------------------------- #
def crystal_system_from_number(spg_number: int) -> str:
    for upper, name in CRYSTAL_SYSTEM_RANGES:
        if spg_number <= upper:
            return name
    raise ValueError(f"invalid space group number {spg_number}")


def symmetry_info(structure: Structure, symprec: float = 0.1, angle_tolerance: float = 5.0) -> dict[str, Any]:
    sga = SpacegroupAnalyzer(structure, symprec=symprec, angle_tolerance=angle_tolerance)
    number = int(sga.get_space_group_number())
    return {
        "space_group_symbol": sga.get_space_group_symbol(),
        "space_group_number": number,
        "crystal_system": crystal_system_from_number(number),
        "lattice_type": sga.get_lattice_type(),
        "point_group": sga.get_point_group_symbol(),
    }


def _stoich_label(counts: dict[str, int]) -> str:
    """AFLOW-style stoichiometry prefix: elements alphabetical -> A, B, C ..."""
    g = reduce(math.gcd, counts.values())
    parts = []
    for i, el in enumerate(sorted(counts)):
        n = counts[el] // g
        parts.append(chr(ord("A") + i) + (str(n) if n != 1 else ""))
    return "".join(parts)


def protostructure_label(structure: Structure, symprec: float = 0.1, include_chemsys: bool = True) -> str:
    """AFLOW-style protostructure label, e.g. ``AB3C_cP5_221_a_c_b:Ca-O-Ti``.

    Mirrors the Matbench-Discovery / aviary convention (stoichiometry, Pearson
    symbol, space group number and per-element Wyckoff sequences). If
    ``aviary`` is installed its implementation (which additionally
    canonicalises equivalent Wyckoff settings) is used instead.
    """
    try:  # pragma: no cover - optional dependency
        from aviary.wren.utils import get_protostructure_label_from_spglib  # type: ignore

        label = get_protostructure_label_from_spglib(structure)
        return label if include_chemsys else label.split(":")[0]
    except Exception:
        pass

    sga = SpacegroupAnalyzer(structure, symprec=symprec)
    sym = sga.get_symmetrized_structure()
    number = int(sga.get_space_group_number())
    hm = sga.get_space_group_symbol()
    system = crystal_system_from_number(number)

    # Wyckoff letters come from the symmetrised input cell (one entry per orbit);
    # multiplicities are taken from the conventional standard cell so that the
    # label is invariant to primitive / conventional / supercell inputs.
    per_element: dict[str, list[str]] = {}
    for sites, wyck in zip(sym.equivalent_sites, sym.wyckoff_symbols):
        per_element.setdefault(sites[0].species_string, []).append(re.sub(r"^\d+", "", wyck))
    conv = sga.get_conventional_standard_structure()
    conv_counts = {str(k): int(round(v)) for k, v in conv.composition.as_dict().items()}

    centering = hm[0]
    if system in {"monoclinic", "orthorhombic"} and centering in {"A", "B"}:
        centering = "C"
    n_conv = sum(conv_counts.values())
    if centering == "R":
        n_conv //= 3  # Pearson symbol for hR counts the rhombohedral cell
    pearson = f"{_PEARSON_FAMILY[system]}{centering}{n_conv}"

    wyckoff_parts = []
    for el in sorted(per_element):
        letters = sorted(per_element[el])
        seq, i = [], 0
        while i < len(letters):
            j = i
            while j < len(letters) and letters[j] == letters[i]:
                j += 1
            run = j - i
            seq.append(letters[i] + (str(run) if run > 1 else ""))
            i = j
        wyckoff_parts.append("".join(seq))

    label = f"{_stoich_label(conv_counts)}_{pearson}_{number}_{'_'.join(wyckoff_parts)}"
    if include_chemsys:
        label += ":" + "-".join(sorted(conv_counts))
    return label


# --------------------------------------------------------------------------- #
# Perovskite helpers (Fig. 5b case study)
# --------------------------------------------------------------------------- #
def vegard_lattice(end_member_params: dict[str, float], fractions: dict[str, float]) -> float:
    """Linear (Vegard) interpolation of a lattice parameter or volume."""
    total = sum(fractions.values())
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        raise ValueError(f"fractions must sum to 1, got {total}")
    missing = set(fractions) - set(end_member_params)
    if missing:
        raise ValueError(f"missing end-member values for {sorted(missing)}")
    return float(sum(end_member_params[k] * f for k, f in fractions.items()))


def shannon_radius(symbol: str, oxidation_state: int, coordination: str) -> float:
    """Shannon ionic radius in Angstrom (pymatgen tables), e.g. ('Cs', 1, 'XII')."""
    return float(Species(symbol, oxidation_state).get_shannon_radius(coordination, radius_type="ionic"))


def perovskite_factors(
    a_site: dict[str, float],
    b_site: dict[str, float],
    x_site: dict[str, float],
    a_ox: int = 1,
    b_ox: int = 2,
    x_ox: int = -1,
    a_cn: str = "XII",
    b_cn: str = "VI",
    x_cn: str = "VI",
) -> dict[str, float]:
    """Goldschmidt tolerance factor t and octahedral factor mu for (mixed) ABX3.

    Site dictionaries map element -> fractional occupancy; radii are occupancy
    weighted (the same averaging used for the CsPb(Cl0.2Br0.4I0.4)3 VCA model).
    """

    def avg(site: dict[str, float], ox: int, cn: str) -> float:
        tot = sum(site.values())
        return sum(shannon_radius(el, ox, cn) * occ for el, occ in site.items()) / tot

    r_a, r_b, r_x = avg(a_site, a_ox, a_cn), avg(b_site, b_ox, b_cn), avg(x_site, x_ox, x_cn)
    t = (r_a + r_x) / (math.sqrt(2) * (r_b + r_x))
    mu = r_b / r_x
    return {"r_A": r_a, "r_B": r_b, "r_X": r_x, "tolerance_factor": t, "octahedral_factor": mu}


def cubic_perovskite(a_el: str, b_el: str, x_occupancy: dict[str, float], a: float) -> Structure:
    """Pm-3m ABX3 cell with (possibly fractional) X-site occupancies."""
    lattice = Lattice.cubic(a)
    species = [{a_el: 1.0}, {b_el: 1.0}, x_occupancy, x_occupancy, x_occupancy]
    coords = [[0.5, 0.5, 0.5], [0, 0, 0], [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]]
    return Structure(lattice, species, coords)


def element_fraction(formula: str | Composition, element: str, among: list[str] | None = None) -> float:
    """Metal-normalised fraction, e.g. V/(V+M) used to rank the NRR candidates."""
    comp = formula if isinstance(formula, Composition) else Composition(formula)
    el = Element(element)
    denom_elements = [Element(e) for e in among] if among else list(comp.elements)
    denom = sum(comp[e] for e in denom_elements)
    return float(comp[el] / denom) if denom else 0.0
