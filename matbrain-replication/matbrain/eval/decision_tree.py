"""Observation-conditioned decision-tree controller (scripted baseline, Fig. 3d-f).

Uses the same Mat-MCP tool pool and budget as the LLM controllers, but the
workflow is hard-coded: each branch is chosen from the previous observation
(e.g. fall back to another candidate when validation fails).
"""

from __future__ import annotations

import json
from typing import Any

from matbrain.mcp.registry import ToolRegistry

MAGNETIC_ELEMENTS = {"V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Ru", "Rh", "Os", "Ir", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "U", "Np", "Pu"}


class DecisionTreeController:
    def __init__(self, registry: ToolRegistry, max_tool_calls: int = 12, efermi_fallback: float = 3.0, eform_fallback: float = -1.5, ehull_fallback: float = 0.1):
        self.registry = registry
        self.max_tool_calls = max_tool_calls
        self.calls = 0
        self.log: list[dict[str, Any]] = []
        self.fallbacks = {"efermi": efermi_fallback, "formation_energy_per_atom": eform_fallback, "energy_above_hull": ehull_fallback}

    async def _call(self, name: str, **args) -> dict | None:
        if self.calls >= self.max_tool_calls or name not in self.registry:
            return None
        self.calls += 1
        res = await self.registry.acall(name, args)
        self.log.append({"tool": name, "arguments": {k: (v if not isinstance(v, str) or len(v) < 200 else v[:200] + "...") for k, v in args.items()}, "ok": res.ok, "error": res.error})
        return res.data if res.ok else None

    @staticmethod
    def _answer(value: Any) -> str:
        return f"<answer>{json.dumps({'value': value})}</answer>"

    async def run(self, task: dict[str, Any]) -> str:
        self.calls, self.log = 0, []
        if task["family"] == "structure_design":
            return await self._design(task["target"])
        return await self._property(task)

    async def _property(self, task: dict[str, Any]) -> str:
        prop = task["property"]
        stored = await self._call("store_structure", cif=task["input_cif"])
        if not stored:
            return self._answer(None)
        h = stored["handle"]
        report = await self._call("validate_structure", cif=h) or {}
        elements = set((report.get("oxidation_states") or {}).keys()) or set()
        if prop == "is_metal":
            gap = await self._call("predict_bandgap", cif=h)
            return self._answer(bool(gap["is_metal"]) if gap else False)
        if prop in ("is_magnetic", "ordering"):
            mag = await self._call("predict_magnetic_moments", cif=h)
            magnetic = bool(mag["is_magnetic"]) if mag else bool(elements & MAGNETIC_ELEMENTS)
            if prop == "is_magnetic":
                return self._answer(magnetic)
            return self._answer("FM" if magnetic else "NM")
        if prop == "formation_energy_per_atom":
            ef = await self._call("predict_formation_energy", cif=h)
            if ef is None:
                ef = await self._call("predict_formation_energy", cif=h, model="M3GNet")
            return self._answer(ef["formation_energy_per_atom"] if ef else self.fallbacks[prop])
        if prop == "energy_above_hull":
            eh = await self._call("phase_diagram_ehull", cif=h, relax=True)
            if eh is None:
                eh = await self._call("phase_diagram_ehull", cif=h, relax=False)
            return self._answer(max(eh["energy_above_hull"], 0.0) if eh else self.fallbacks[prop])
        if prop == "efermi":
            v = await self._call("predict_property_custom", cif=h, property="efermi")
            return self._answer(v["value"] if v else self.fallbacks[prop])
        return self._answer(None)

    async def _design(self, target: dict[str, Any]) -> str:
        gen = await self._call("crystallm_generate", formula=target["formula"], spacegroup=target["spacegroup_symbol"], num_samples=4)
        candidates = [r["handle"] for r in (gen or {}).get("results", []) if r.get("handle")]
        if not candidates:
            # fallback branch: unconditioned generation
            gen = await self._call("crystallm_generate", formula=target["formula"], num_samples=4)
            candidates = [r["handle"] for r in (gen or {}).get("results", []) if r.get("handle")]
        best, best_score = None, -1
        for h in candidates:
            rep = await self._call("validate_structure", cif=h)
            if not rep:
                continue
            score = sum([
                rep.get("valid", False),
                rep.get("charge_balanced", False),
                rep.get("num_sites") == target["nsites"],
                rep.get("space_group_number") == target["spacegroup_number"],
                rep.get("crystal_system") == target["crystal_system"],
            ])
            if score > best_score:
                best, best_score = h, score
            if score == 5:
                break
        if best is None:
            return "<answer></answer>"
        cif = await self._call("get_structure_cif", handle=best)
        return f"<answer>{cif['cif'] if cif else best}</answer>"
