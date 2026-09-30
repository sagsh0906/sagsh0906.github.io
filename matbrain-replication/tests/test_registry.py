import asyncio
import json

from pymatgen.core import Composition

from matbrain import chem
from matbrain.mcp.artifacts import get_store
from matbrain.mcp.server import build_server


def test_registry_contents(registry):
    names = set(registry.names())
    for expected in ["mp_search", "oqmd_search", "crystallm_generate", "mattergen_generate", "validate_structure", "phase_diagram_ehull", "relax_structure", "predict_bandgap", "web_search", "literature_search", "vasp_prepare_inputs"]:
        assert expected in names
    pool = registry.subset(disable_target_db=True)
    assert "mp_search" not in pool and "oqmd_search" not in pool and "validate_structure" in pool


def test_validation_layers(registry, nacl):
    assert not registry.validate("does_not_exist", {}).ok
    assert "not valid JSON" in registry.validate("validate_structure", "{bad json").error
    assert "Extra inputs" in registry.validate("check_charge_balance", {"formula": "NaCl", "x": 1}).error
    assert not registry.validate("validate_structure", {"cif": "garbage"}).ok
    assert registry.validate("validate_structure", {"cif": chem.structure_to_cif(nacl)}).ok
    # handle format is accepted offline, existence is enforced in strict mode
    assert registry.validate("validate_structure", {"cif": "cif://0123456789ab"}).ok
    assert not registry.validate("validate_structure", {"cif": "cif://0123456789ab"}, strict_handles=True).ok
    h = get_store().put_structure(nacl)
    assert registry.validate("validate_structure", {"cif": h}, strict_handles=True).ok


def test_flat_schema_for_verl(registry):
    for spec in registry:
        schema = spec.openai_schema(flat=True)["function"]["parameters"]
        for prop in schema["properties"].values():
            assert set(prop) <= {"type", "description", "enum"}


def test_tool_calls(registry, nacl):
    res = registry.call("validate_structure", {"cif": chem.structure_to_cif(nacl)})
    assert res.ok and res.data["charge_balanced"] and res.data["space_group_number"] == 225
    res = registry.call("crystallm_generate", {"formula": "CsPbCl0.6Br1.2I1.2"})
    assert not res.ok and "Non-stoichiometric" in res.error
    res = registry.call("normalize_composition", {"formula": "CsPb(Cl0.2Br0.4I0.4)3"})
    assert Composition(res.data["output"]) == Composition("Cs5Pb5Cl3Br6I6")
    res = registry.call("phase_diagram_ehull", {"formula": "NaCl"})
    assert not res.ok and "provide either" in res.error


def test_dedup_tool(registry, nacl):
    cifs = [chem.structure_to_cif(nacl), chem.structure_to_cif(nacl.get_primitive_structure()), chem.structure_to_cif(nacl * (1, 1, 2))]
    res = registry.call("deduplicate_structures", {"cifs": cifs})
    assert res.ok and res.data["n_unique"] == 1


def test_mcp_server_roundtrip(registry):
    srv = build_server(registry)

    async def go():
        tools = await srv.list_tools()
        assert len(tools) == len(registry)
        out = await srv.call_tool("vegard_interpolate", {"end_member_values": {"a": 5.0, "b": 6.0}, "fractions": {"a": 0.5, "b": 0.5}})
        return json.loads(out.content[0].text)

    payload = asyncio.run(go())
    assert payload["status"] == "success" and payload["result"]["value"] == 5.5


def test_flat_schemas_for_verl_mcp_client(registry):
    """verl's MCP client validates every property against {type, description, enum}."""
    srv = build_server(registry, flat_schemas=True)
    tools = asyncio.run(srv.list_tools())
    for t in tools:
        schema = t.input_schema if hasattr(t, "input_schema") else t.inputSchema
        for prop in schema["properties"].values():
            assert "type" in prop, (t.name, prop)
    # arguments are still validated by the Pydantic models
    out = asyncio.run(srv.call_tool("check_charge_balance", {"formula": "V2Fe6S4"}))
    assert '"charge_balanced": false' in out.content[0].text
