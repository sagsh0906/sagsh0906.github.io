import json
from datetime import datetime, timedelta

import pytest
from pymatgen.core import Lattice, Structure

from matbrain.chem import structure_to_cif
from matbrain.data.benchmark import build_tasks
from matbrain.data.generate_distil import consistent, mp_questions, parse_json_list, parse_score, passes
from matbrain.data.leakage import AuditConfig, audit, summarize
from matbrain.data.literature import majority_vote, parse_category
from matbrain.data.rl_dataset import to_rl_row
from matbrain.data.split import timestamp_split
from matbrain.mcp.artifacts import ArtifactStore


def rec(mid, s, **kw):
    return {"material_id": mid, "formula": s.composition.reduced_formula, "cif": structure_to_cif(s), **kw}


def rocksalt(a, b, lat):
    return Structure.from_spacegroup("Fm-3m", Lattice.cubic(lat), [a, b], [[0, 0, 0], [0.5, 0.5, 0.5]])


def test_timestamp_split():
    t0 = datetime(2020, 1, 1)
    rows = [{"material_id": f"mp-{i}", "created_at": (t0 + timedelta(days=i)).isoformat()} for i in range(20)]
    rows += [{"material_id": "mp-x", "created_at": None}]
    sp = timestamp_split(rows, test_size=5, val_size=3)
    assert [len(sp[k]) for k in ("train", "val", "test", "excluded_no_timestamp")] == [12, 3, 5, 1]
    assert sp["test"][-1]["material_id"] == "mp-19" and sp["val"][0]["material_id"] == "mp-12"


def test_leakage_audit_levels():
    nacl = rocksalt("Na", "Cl", 5.64)
    train = [rec("mp-1", nacl), rec("mp-2", rocksalt("Mg", "O", 4.21))]
    test = [
        rec("mp-1", rocksalt("Na", "Cl", 5.70)),                 # identity leak
        {**rec("mp-10", nacl)},                                   # exact CIF duplicate
        rec("mp-11", nacl.get_primitive_structure()),            # same structure, different cell
        rec("mp-12", Structure(Lattice.cubic(3.0), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])),  # NaCl formula, CsCl-type
        rec("mp-13", rocksalt("K", "Br", 6.6)),                   # new chemistry, same prototype w/o chemsys
    ]
    flags = {f.material_id: f for f in audit(train, test, AuditConfig(workers=1))}
    assert flags["mp-1"].mp_id and flags["mp-1"].excluded
    assert flags["mp-10"].cif_hash and flags["mp-10"].excluded
    assert flags["mp-11"].structure_match and flags["mp-11"].protostructure and not flags["mp-11"].cif_hash
    assert flags["mp-12"].formula and not flags["mp-12"].excluded  # formula overlap is a flag only
    assert not flags["mp-13"].excluded and not flags["mp-13"].formula
    rows = summarize(flags.values())
    assert rows[-1]["n_overlap"] == 3


def test_benchmark_tasks():
    s = rocksalt("Na", "Cl", 5.64)
    r = rec("mp-5", s, spacegroup_symbol="Fm-3m", spacegroup_number=225, crystal_system="cubic", is_metal=False, is_magnetic=False, ordering="NM", formation_energy_per_atom=-2.1, energy_above_hull=0.0, efermi=1.2)
    tasks = build_tasks([r])
    fams = sorted({t["family"] for t in tasks})
    assert fams == ["classification", "regression", "structure_design"]
    design = next(t for t in tasks if t["family"] == "structure_design")
    assert design["target"]["nsites"] == 8 and "mp-5" not in design["prompt"]
    assert all("mp-5" not in t["prompt"] for t in tasks)
    unknown = build_tasks([{**r, "ordering": "Unknown"}], properties=["ordering"], design=False)
    assert unknown == []


def test_generate_distil_helpers():
    assert parse_json_list('<think>x</think>["What is the band gap of GaN?", "short"]') == ["What is the band gap of GaN?"]
    assert parse_score("blah <score>5</score>") == 5.0 and parse_score("4/5") == 4.0
    assert passes(5, 4) and not passes(4, 4) and passes(4, 4, strict=False)
    s = rocksalt("Na", "Cl", 5.64)
    items = mp_questions(rec("mp-1", s, formation_energy_per_atom=-2.0, spacegroup_symbol="Fm-3m", spacegroup_number=225, crystal_system="cubic", nsites=8, ordering="Unknown"))
    tasks = {i["task"] for i in items}
    assert "property:formation_energy_per_atom" in tasks and "structure_generation" in tasks and "property:ordering" not in tasks
    assert consistent("property:formation_energy_per_atom", '{"value": -1.95}', "-2.000")
    assert not consistent("property:formation_energy_per_atom", '{"value": -1.2}', "-2.000")
    assert consistent("structure_generation", structure_to_cif(s.get_primitive_structure()), structure_to_cif(s))


def test_majority_vote():
    assert majority_vote(["Synthesis", "Synthesis", "Structure"]) == "Synthesis"
    assert majority_vote(["Synthesis", "Properties", "Structure"]) is None
    assert majority_vote(["Synthesis", "Properties", "Structure"], tie_break="first") == "Synthesis"
    assert parse_category("<think>...</think> Applications") == "Applications"


def test_rl_row(tmp_path):
    s = rocksalt("Na", "Cl", 5.64)
    item = {"messages": [{"role": "user", "content": "What is the formation energy per atom?\n\n" + structure_to_cif(s)}, {"role": "assistant", "content": "x"}], "task": "property:formation_energy_per_atom", "material_id": "mp-1"}
    store = ArtifactStore(tmp_path)
    row = to_rl_row(item, 0, "train", store)
    handle = row["extra_info"]["prompt_text"].split("stored as ")[1].split(".")[0]
    assert store.exists(handle) and row["agent_name"] == "tool_agent"
    assert row["prompt"][0]["role"] == "system" and json.dumps(row)
    with pytest.raises(KeyError):
        ArtifactStore().get(handle)
