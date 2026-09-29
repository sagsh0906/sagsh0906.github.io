import math

import pytest
from pymatgen.core import Composition, Lattice, Structure

from matbrain import chem


def test_vegard_matches_paper_fig5b():
    a = chem.vegard_lattice({"Cl": 5.67984, "Br": 5.94863, "I": 6.27514}, {"Cl": 0.2, "Br": 0.4, "I": 0.4})
    assert a == pytest.approx(6.025476, abs=1e-6)
    v = chem.vegard_lattice({"Cl": 183.235, "Br": 210.499, "I": 247.099}, {"Cl": 0.2, "Br": 0.4, "I": 0.4})
    assert v == pytest.approx(219.687, abs=1e-3)


def test_perovskite_factors_match_paper():
    f = chem.perovskite_factors({"Cs": 1}, {"Pb": 1}, {"Cl": 0.2, "Br": 0.4, "I": 0.4})
    assert round(f["tolerance_factor"], 3) == 0.859
    assert round(f["octahedral_factor"], 3) == 0.587


def test_vca_cell_volume_and_symmetry():
    s = chem.cubic_perovskite("Cs", "Pb", {"Cl": 0.2, "Br": 0.4, "I": 0.4}, 6.025476)
    assert s.volume == pytest.approx(218.76, abs=0.01)
    assert chem.symmetry_info(s)["space_group_symbol"] == "Pm-3m"
    assert chem.charge_balance(s.composition)["charge_balanced"]


def test_integer_approximant():
    comp = chem.integerize_composition(Composition("CsPbCl0.6Br1.2I1.2"))
    assert comp == Composition("Cs5Pb5Cl3Br6I6")


def test_valence_filter_examples():
    assert not chem.charge_balance("V2Fe6S4")["charge_balanced"]  # paper's rejected example
    assert chem.charge_balance("CoV4S8")["charge_balanced"]
    assert chem.charge_balance("Fe")["charge_balanced"]


def test_protostructure_label_invariant_to_cell_choice(nacl):
    label = chem.protostructure_label(nacl)
    assert label == "AB_cF8_225_b_a:Cl-Na"
    assert chem.protostructure_label(nacl.get_primitive_structure()) == label
    assert chem.protostructure_label(nacl * (2, 1, 1)) == label
    kcl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(6.29), ["K", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    assert chem.protostructure_label(kcl, include_chemsys=False) == chem.protostructure_label(nacl, include_chemsys=False)
    assert chem.protostructure_label(kcl) != label


def test_rhombohedral_pearson():
    bi = Structure.from_spacegroup("R-3m", Lattice.hexagonal(4.546, 11.86), ["Bi"], [[0, 0, 0.2339]])
    assert chem.protostructure_label(bi) == "A_hR2_166_c:Bi"


def test_validity_and_hash(nacl):
    assert chem.structural_validity(nacl)["valid"]
    bad = Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [0.01, 0, 0]])
    assert not chem.structural_validity(bad)["valid"]
    cif = chem.structure_to_cif(nacl)
    assert chem.cif_hash(cif) == chem.cif_hash("# comment\n" + cif.replace("\n", "\n   "))
    assert chem.parse_structure(cif).composition.reduced_formula == "NaCl"
    with pytest.raises(chem.StructureParseError):
        chem.parse_structure("data_x\n_cell_length_a 1\n")


def test_element_fraction():
    assert chem.element_fraction("CoV4S8", "V", ["V", "Co"]) == pytest.approx(0.8)
    assert math.isclose(chem.element_fraction("FeV2S4", "V", ["V", "Fe"]), 2 / 3)
