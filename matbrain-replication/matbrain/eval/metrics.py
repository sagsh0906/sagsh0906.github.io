"""Benchmark metrics (paper Fig. 3).

Classification: balanced accuracy (metallic, magnetic), macro-F1 (magnetic
order). Regression: MAE. Structure design: seven rates - valid structure,
charge-balanced, formula, atom count, crystal system and space-group
consistency, and complete success (all six satisfied).

Unparseable / missing predictions count as wrong (classification) and are
reported separately for regression (MAE over parsed predictions plus the
coverage), so that systems cannot improve MAE by abstaining silently.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

from pymatgen.core import Composition

from matbrain import chem

DESIGN_METRICS = ("complete_success", "valid_structure", "charge_balanced", "formula", "atom_count", "crystal_system", "space_group")


# --------------------------------------------------------------------------- #
# Answer parsing
# --------------------------------------------------------------------------- #
def _strip(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.findall(r"<answer>(.*?)</answer>", text, re.S)
    return (m[-1] if m else text).strip()


def parse_value(text: str) -> Any:
    body = _strip(text)
    m = re.search(r"\{[^{}]*\"value\"[^{}]*\}", body, re.S)
    if m:
        try:
            return json.loads(m.group(0))["value"]
        except json.JSONDecodeError:
            pass
    m = re.search(r'"?value"?\s*[:=]\s*"?([^",}\n]+)', body)
    return (m.group(1) if m else body).strip()


def parse_bool(text: str) -> bool | None:
    v = parse_value(text)
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if re.match(r"^(true|yes|metallic|magnetic)\b", s):
        return True
    if re.match(r"^(false|no|non-?metallic|non-?magnetic|not)\b", s):
        return False
    return None


def parse_ordering(text: str) -> str | None:
    s = str(parse_value(text)).strip()
    for canon, pats in {"FiM": [r"\bfim\b", r"ferri"], "AFM": [r"\bafm\b", r"antiferro"], "FM": [r"\bfm\b", r"(?<!anti)(?<!ferri)ferromag"], "NM": [r"\bnm\b", r"non-?magnetic"]}.items():
        if any(re.search(p, s, re.I) for p in pats):
            return canon
    return None


def parse_float(text: str) -> float | None:
    v = parse_value(text)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return float(v)
    m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(v).replace("−", "-"))
    return float(m.group(0)) if m else None


def parse_cif(text: str) -> str | None:
    body = _strip(text)
    body = re.sub(r"^```\w*\n|```$", "", body.strip(), flags=re.M)
    m = re.search(r"(data_.*)", body, re.S)
    return m.group(1).strip() if m else None


# --------------------------------------------------------------------------- #
# Aggregate metrics
# --------------------------------------------------------------------------- #
def balanced_accuracy(y_true: list[bool], y_pred: list[bool | None]) -> float:
    recalls = []
    for cls in (True, False):
        idx = [i for i, y in enumerate(y_true) if y == cls]
        if idx:
            recalls.append(sum(y_pred[i] == cls for i in idx) / len(idx))
    return sum(recalls) / len(recalls) if recalls else float("nan")


def macro_f1(y_true: list[str], y_pred: list[str | None], labels: tuple[str, ...] = ("NM", "FM", "FiM", "AFM")) -> float:
    f1s = []
    for c in labels:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        if tp + fp + fn == 0:
            continue  # class absent from both labels and predictions
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return sum(f1s) / len(f1s) if f1s else float("nan")


def mae(y_true: list[float], y_pred: list[float | None]) -> tuple[float, float]:
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None and math.isfinite(p)]
    if not pairs:
        return float("nan"), 0.0
    return sum(abs(t - p) for t, p in pairs) / len(pairs), len(pairs) / len(y_true)


def design_scores(cif: str | None, target: dict[str, Any], symprec: float = 0.1) -> dict[str, bool]:
    out = {k: False for k in DESIGN_METRICS}
    if not cif:
        return out
    try:
        s = chem.parse_structure(cif)
    except chem.StructureParseError:
        return out
    out["valid_structure"] = chem.structural_validity(s)["valid"]
    out["charge_balanced"] = chem.charge_balance(s.composition)["charge_balanced"]
    out["formula"] = s.composition.reduced_formula == Composition(target["formula"]).reduced_formula
    out["atom_count"] = len(s) == int(target["nsites"])
    try:
        sym = chem.symmetry_info(s, symprec=symprec)
        out["crystal_system"] = sym["crystal_system"] == str(target["crystal_system"]).lower()
        out["space_group"] = sym["space_group_number"] == int(target["spacegroup_number"])
    except Exception:
        pass
    out["complete_success"] = all(out[k] for k in DESIGN_METRICS if k != "complete_success")
    return out


def score_predictions(tasks: list[dict], outputs: dict[str, str]) -> dict[str, Any]:
    """tasks: benchmark rows; outputs: task id -> raw model output."""
    by_prop: dict[str, list[tuple[dict, str | None]]] = {}
    for t in tasks:
        by_prop.setdefault(t["property"], []).append((t, outputs.get(t["id"])))
    report: dict[str, Any] = {}
    for prop, rows in by_prop.items():
        if prop in ("is_metal", "is_magnetic"):
            report[prop] = {"balanced_accuracy": balanced_accuracy([t["label"] for t, _ in rows], [parse_bool(o) if o else None for _, o in rows]), "n": len(rows)}
        elif prop == "ordering":
            report[prop] = {"macro_f1": macro_f1([t["label"] for t, _ in rows], [parse_ordering(o) if o else None for _, o in rows]), "n": len(rows)}
        elif prop == "structure":
            scores = [design_scores(parse_cif(o) if o else None, t["target"]) for t, o in rows]
            report["structure_design"] = {k: sum(s[k] for s in scores) / len(scores) for k in DESIGN_METRICS} | {"n": len(rows)}
        else:
            err, cov = mae([t["label"] for t, _ in rows], [parse_float(o) if o else None for _, o in rows])
            report[prop] = {"mae": err, "coverage": cov, "n": len(rows)}
    return report
