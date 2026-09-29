import math

import numpy as np
import pytest
from pymatgen.core import Composition, Lattice, Structure

from matbrain.analysis.entropy import adjacent_average, entropy_from_top_logprobs, kde_curve, kde_peaks, segment_entropy
from matbrain.chem import structure_to_cif
from matbrain.eval.metrics import balanced_accuracy, design_scores, macro_f1, mae, parse_bool, parse_cif, parse_float, parse_ordering, score_predictions
from matbrain.experiments import nrr
from matbrain.screening.funnel import FunnelConfig, ScreeningFunnel


# ------------------------------------------------------------------ metrics
def test_classification_metrics():
    assert balanced_accuracy([True, True, False, False], [True, False, False, False]) == pytest.approx(0.75)
    assert balanced_accuracy([True, False], [None, False]) == pytest.approx(0.5)
    assert macro_f1(["NM", "FM", "AFM", "NM"], ["NM", "FM", "FM", None]) == pytest.approx((2 / 3 + 2 / 3 + 0) / 3)
    err, cov = mae([1.0, 2.0, 3.0], [1.5, None, 2.0])
    assert err == pytest.approx(0.75) and cov == pytest.approx(2 / 3)


def test_answer_parsers():
    assert parse_bool('<think>..</think><answer>{"value": true}</answer>') is True
    assert parse_bool("<answer>Non-magnetic</answer>") is False
    assert parse_ordering('<answer>{"value": "FiM"}</answer>') == "FiM"
    assert parse_ordering("<answer>antiferromagnetic</answer>") == "AFM"
    assert parse_ordering("<answer>ferromagnetic</answer>") == "FM"
    assert parse_float('<answer>{"value": -1.234}</answer>') == pytest.approx(-1.234)
    assert parse_float("<answer>about −0.52 eV/atom</answer>") == pytest.approx(-0.52)


def test_design_scores(nacl):
    target = {"formula": "Na4Cl4", "nsites": 8, "crystal_system": "cubic", "spacegroup_number": 225}
    ok = design_scores(parse_cif(f"<answer>```\n{structure_to_cif(nacl)}\n```</answer>"), target)
    assert all(ok.values())
    prim = design_scores(structure_to_cif(nacl.get_primitive_structure()), target)
    assert prim["formula"] and prim["space_group"] and not prim["atom_count"] and not prim["complete_success"]
    assert not any(design_scores(None, target).values())


def test_score_predictions(nacl):
    tasks = [
        {"id": "a", "property": "is_metal", "label": False},
        {"id": "b", "property": "formation_energy_per_atom", "label": -2.0},
        {"id": "c", "property": "structure", "target": {"formula": "NaCl", "nsites": 8, "crystal_system": "cubic", "spacegroup_number": 225}},
    ]
    rep = score_predictions(tasks, {"a": '<answer>{"value": false}</answer>', "b": '<answer>{"value": -1.9}</answer>', "c": f"<answer>{structure_to_cif(nacl)}</answer>"})
    assert rep["is_metal"]["balanced_accuracy"] == 1.0
    assert rep["formation_energy_per_atom"]["mae"] == pytest.approx(0.1)
    assert rep["structure_design"]["complete_success"] == 1.0


# ------------------------------------------------------------------ entropy
def test_adjacent_average():
    x = np.arange(10, dtype=float)
    y = adjacent_average(x, 3)
    assert y[0] == pytest.approx(0.5) and y[5] == pytest.approx(5.0) and y[-1] == pytest.approx(8.5)


def test_topk_entropy_and_segments():
    uniform4 = {str(i): math.log(0.25) for i in range(4)}
    assert entropy_from_top_logprobs([uniform4]) == [pytest.approx(2.0)]
    toks = ["<think>", "a", "</think>", "<answer>", "b", "</answer>"]
    seg = segment_entropy(toks, [1, 1, 1, 0.1, 0.1, 0.1])
    assert seg["think"]["mean"] == pytest.approx(1.0) and seg["answer"]["mean"] == pytest.approx(0.1)


def test_kde_bimodal_peak_and_scaling():
    rng = np.random.default_rng(0)
    vals = np.concatenate([rng.exponential(0.3, 5000), rng.normal(3.3, 0.15, 800)])
    c = kde_curve(vals)
    assert c["y"].max() == pytest.approx(c["hist_counts"].max())
    assert c["x"][0] < vals.min() - 4 * c["bandwidth"]
    assert any(abs(p - 3.3) < 0.2 for p in kde_peaks(c))


# ------------------------------------------------------------------ screening
def _spinel(m):
    """Cubic spinel MV2S4 (Fd-3m, origin choice 1: 8a, 16d, 32e)."""
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(10.0), [m, "V", "S"], [[0, 0, 0], [0.625, 0.625, 0.625], [0.385, 0.385, 0.385]])


def test_screening_funnel_stages():
    co, fe, ni = _spinel("Co"), _spinel("Fe"), _spinel("Ni")
    binary = Structure.from_spacegroup("Pa-3", Lattice.cubic(5.5), ["Co", "S"], [[0, 0, 0], [0.39, 0.39, 0.39]])
    impure = co.copy()
    impure.replace(0, "O")
    unbalanced = Structure(Lattice.cubic(8), ["V", "V", "Fe", "Fe", "Fe", "Fe", "Fe", "Fe", "S", "S", "S", "S"], [[i / 12, i / 12, i / 12] for i in range(12)])
    overlap = co.copy()
    overlap.translate_sites([0], [0.03, 0, 0])  # breaks symmetry, keeps CoV2S4 ...
    overlap.append("S", overlap[1].frac_coords + [0.01, 0, 0])  # ... but adds an S atom 0.1 A from V
    overlap.remove_sites([len(overlap) - 2])  # keep the composition charge balanced
    structures = [co, co.copy(), fe, ni, binary, impure, unbalanced, overlap]
    rf = lambda f: Composition(f).reduced_formula  # noqa: E731
    ehull = {rf("CoV2S4"): 0.0, rf("FeV2S4"): 0.01, rf("NiV2S4"): 0.2}
    gaps = {rf("CoV2S4"): 0.9, rf("FeV2S4"): 1.5}
    funnel = ScreeningFunnel(
        FunnelConfig(),
        ehull_fn=lambda ss: [ehull[s.composition.reduced_formula] for s in ss],
        bandgap_fn=lambda s: gaps.get(s.composition.reduced_formula),
        reference_fn=lambda chemsys: [fe] if chemsys == "Fe-S-V" else [],
    )
    ranked = funnel.run(structures)
    counts = [(s.name[:2], s.n_in, s.n_out) for s in funnel.stages]
    assert counts == [("01", 8, 6), ("02", 6, 5), ("03", 5, 4), ("04", 4, 3), ("05", 3, 2), ("06", 2, 2), ("07", 2, 1)]
    assert [c.props["formula"] for c in ranked] == [rf("CoV2S4")]
    assert ranked[0].props["V_fraction"] == pytest.approx(2 / 3, abs=1e-4)


# ------------------------------------------------------------------ NRR
def test_nrr_equations():
    assert nrr.e_rhe(-0.747, ph=1.0) == pytest.approx(-0.491)
    c = nrr.concentration_from_absorbance(0.4417 * 0.2 + 0.0305)
    assert c == pytest.approx(0.2)
    assert nrr.nh3_yield(c) == pytest.approx(0.2 * 30 / (0.1 * 2))  # 30 ug h-1 mg-1
    q = 3 * 96485 * (0.2 * 30e-6) / 17 / 0.05  # charge giving FE = 5 %
    assert nrr.faradaic_efficiency(0.2, q) == pytest.approx(0.05)
