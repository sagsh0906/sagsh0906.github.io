"""Electrochemical NRR data reduction (paper Methods, Eqs. 6-8).

    E_RHE    = E_Ag/AgCl + 0.059 pH + 0.197                       (6)
    NH3 yield = C_NH3 * V / (m_cat * t)                            (7)
    FE       = 3 F C_NH3 V / (17 Q)                                (8)

Defaults follow the paper: catholyte V = 30 ml, catalyst loading 0.1 mg on
1x1 cm2 carbon paper, electrolysis time 2 h, 0.1 M HCl (pH ~ 1), indophenol
calibration A = 0.4417 C + 0.0305 (C in ug/ml, R^2 = 0.999, Fig. 1c).

    python -m matbrain.experiments.nrr --absorbance 0.21 --charge 1.3 --e-ag-agcl -0.608
"""

from __future__ import annotations

import argparse
import json

FARADAY = 96485.0  # C/mol
M_NH3 = 17.0  # g/mol (relative molecular mass used in the paper)


def e_rhe(e_ag_agcl: float, ph: float = 1.0) -> float:
    return e_ag_agcl + 0.059 * ph + 0.197


def e_ag_agcl_for(e_rhe_target: float, ph: float = 1.0) -> float:
    return e_rhe_target - 0.059 * ph - 0.197


def concentration_from_absorbance(absorbance: float, slope: float = 0.4417, intercept: float = 0.0305, dilution: float = 1.0) -> float:
    """NH3 concentration (ug/ml) from the indophenol-blue calibration line."""
    return (absorbance - intercept) / slope * dilution


def nh3_yield(c_ug_per_ml: float, volume_ml: float = 30.0, m_cat_mg: float = 0.1, hours: float = 2.0) -> float:
    """NH3 yield rate in ug h^-1 mg_cat^-1."""
    return c_ug_per_ml * volume_ml / (m_cat_mg * hours)


def faradaic_efficiency(c_ug_per_ml: float, charge_c: float, volume_ml: float = 30.0, n_electrons: int = 3) -> float:
    """Faradaic efficiency (fraction) for NH3; C in ug/ml, V in ml, Q in coulomb."""
    mass_g = c_ug_per_ml * volume_ml * 1e-6
    return n_electrons * FARADAY * mass_g / (M_NH3 * charge_c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--absorbance", type=float, help="indophenol absorbance (~655 nm)")
    ap.add_argument("--concentration", type=float, help="NH3 concentration in ug/ml (instead of absorbance)")
    ap.add_argument("--charge", type=float, required=True, help="charge passed Q (C)")
    ap.add_argument("--volume", type=float, default=30.0)
    ap.add_argument("--mass", type=float, default=0.1, help="catalyst mass (mg)")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--e-ag-agcl", type=float)
    ap.add_argument("--ph", type=float, default=1.0)
    args = ap.parse_args()
    c = args.concentration if args.concentration is not None else concentration_from_absorbance(args.absorbance)
    out = {
        "c_nh3_ug_per_ml": c,
        "nh3_yield_ug_per_h_per_mg": nh3_yield(c, args.volume, args.mass, args.hours),
        "faradaic_efficiency_percent": 100 * faradaic_efficiency(c, args.charge, args.volume),
    }
    if args.e_ag_agcl is not None:
        out["e_rhe"] = e_rhe(args.e_ag_agcl, args.ph)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
