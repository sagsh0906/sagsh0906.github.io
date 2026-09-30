"""Mean-field crystal-plasticity proxy used in place of the paper's CPFE model (virtual tensile test).

The paper validates generated microstructures with a calibrated CPFE model (DAMASK-like, {111}<110> slip,
dislocation-density hardening, grain-boundary strengthening, damage) that takes Euler angles and a learned GND map as
input.  A full CPFE solve is out of reach here, so this module implements a much lighter, but still
microstructure-resolved, grain-level model driven by the same inputs extracted from an Euler-map image:

* per-grain Schmid factor m_g of the 12 {111}<110> systems (load along RD = image x) -> Sachs factor M_g = 1/m_g;
* per-grain size d_g (from the segmented grains) -> Hall-Petch CRSS  tau_c = tau_0 + k_HP / sqrt(d_g);
* per-grain initial dislocation density from the kernel average misorientation (GND = 2 theta / (u b));
* Kocks-Mecking-Estrin evolution  d rho / d gamma = (1/b)(1/(beta d_g) + k1 sqrt(rho)) - k2 rho + HDI term;
  the HDI term adds geometrically necessary dislocations at interfaces between grains of very different size
  (hetero-deformation induced hardening, the mechanism the paper invokes for bimodal microstructures);
* iso-strain (Taylor/Voigt) homogenisation, 0.2 % yield strength, Considere criterion for uniform elongation and a
  constant post-uniform (necking) strain for the elongation to failure.

It is a *proxy*: parameters are order-of-magnitude values for Inconel 625, calibrated so that the 25 synthetic
microstructures give yield strengths / elongations in the measured range.  Use it for relative comparisons.
"""
from dataclasses import dataclass

import numpy as np
from skimage.segmentation import expand_labels

from . import euler as E
from .features import grain_table, kam_map, segment_grains, PIXEL_UM


@dataclass
class CPParams:
    shear_modulus: float = 79e3        # MPa
    burgers: float = 2.54e-10          # m
    alpha: float = 0.3                 # Taylor constant
    tau0: float = 110.0                # MPa, lattice friction + solid solution (resolved)
    k_hp: float = 0.34                 # MPa m^0.5, resolved Hall-Petch coefficient
    k1: float = 0.03                   # forest storage coefficient
    k2: float = 1.2                    # dynamic recovery coefficient
    beta: float = 15.0                 # grain-boundary storage coefficient
    k_hdi: float = 0.05                # hetero-deformation (size-mismatch interface) GND storage
    size_ratio_hdi: float = 3.0        # neighbour size ratio regarded as a hetero-interface
    rho_floor: float = 1e12            # m^-2, annealed dislocation density
    kam_floor_deg: float = 0.15        # KAM noise floor subtracted before converting to GND density
    kam_scale: float = 1.0             # KAM -> GND calibration constant
    kam_max_deg: float = 8.0           # KAM kernel threshold (larger misorientations = boundaries)
    lamella_aspect: float = 3.0        # thin (twin) lamellae are not counted as fine grains for HDI
    post_uniform: float = 0.22         # engineering strain added after necking onset
    eps_max: float = 1.2
    n_steps: int = 2400


def grain_inputs(img, params=CPParams(), pixel_um=PIXEL_UM):
    """Extract per-grain inputs (area fraction, size, Sachs factor, initial rho, hetero-interface fraction)."""
    labels = segment_grains(img)
    labels_full = expand_labels(labels, distance=3)
    t = grain_table(img, labels=labels_full, pixel_um=pixel_um)
    n = len(t['label'])
    if n == 0:
        raise ValueError('no grains found in image')
    area = t['area']
    d = np.sqrt(4 * area / np.pi) * 1e-6                                 # m
    eul = np.radians(np.stack([t['euler1'], t['euler2'], t['euler3']], 1))
    m = E.schmid_factor_fcc(E.euler_to_matrix(eul))
    sachs = 1.0 / np.clip(m, 0.2, 0.5)
    # initial dislocation density from KAM (grain average)
    kam, _ = kam_map(img, labels=labels, max_deg=params.kam_max_deg)
    lut = np.zeros(labels_full.max() + 1, int)
    lut[t['label']] = np.arange(n)
    kam_g = np.zeros(n)
    valid = np.isfinite(kam) & (labels > 0)
    if valid.any():
        idx = lut[labels[valid]]
        s = np.bincount(idx, weights=kam[valid], minlength=n)
        c = np.bincount(idx, minlength=n)
        kam_g = np.where(c > 0, s / np.maximum(c, 1), 0.0)
    theta = np.radians(np.clip(kam_g - params.kam_floor_deg, 0.0, None))
    rho0 = params.rho_floor + params.kam_scale * 2 * theta / (pixel_um * 1e-6 * params.burgers)
    # fraction of each grain's boundary shared with grains of very different size
    pair_len = {}
    for a, b in ((labels_full[:, :-1], labels_full[:, 1:]), (labels_full[:-1, :], labels_full[1:, :])):
        msk = (a != b) & (a > 0) & (b > 0)
        ia, ib = lut[a[msk]], lut[b[msk]]
        for x, y in zip(ia.tolist(), ib.tolist()):
            key = (x, y) if x < y else (y, x)
            pair_len[key] = pair_len.get(key, 0) + 1
    het = np.zeros(n)
    tot = np.zeros(n)
    lamella = t['major_axis'] > params.lamella_aspect * np.maximum(t['minor_axis'], 1e-9)
    for (x, y), L in pair_len.items():
        small = x if d[x] < d[y] else y
        ratio = max(d[x], d[y]) / min(d[x], d[y])
        h = float(ratio > params.size_ratio_hdi and not lamella[small])
        het[x] += h * L
        het[y] += h * L
        tot[x] += L
        tot[y] += L
    het = np.where(tot > 0, het / np.maximum(tot, 1), 0.0)
    w = area / area.sum()
    return dict(w=w, d=d, sachs=sachs, rho0=rho0, het=het, kam=kam_g)


def tensile_curve(inputs, params=CPParams()):
    """Integrate the grain-level model; returns true strain / true stress and engineering curves."""
    p = params
    w, d, M, rho, het = (inputs[k].astype(float) for k in ('w', 'd', 'sachs', 'rho0', 'het'))
    rho = rho.copy()
    eps = np.linspace(0.0, p.eps_max, p.n_steps + 1)
    de = eps[1] - eps[0]
    tau_c = p.tau0 + p.k_hp / np.sqrt(d)
    taylor = p.alpha * p.shear_modulus * p.burgers
    sig = np.empty_like(eps)
    for i in range(len(eps)):
        sig[i] = np.sum(w * M * (tau_c + taylor * np.sqrt(rho)))
        dgam = M * de
        drho =(1.0 / p.burgers) * (1.0 / (p.beta * d) + p.k1 * np.sqrt(rho)) - p.k2 * rho \
            + p.k_hdi * het / (p.burgers * d)
        rho = np.maximum(rho + drho * dgam, p.rho_floor)
    return eps, sig


def properties(img, params=CPParams(), pixel_um=PIXEL_UM, return_curve=False):
    """Yield strength (MPa, 0.2 % plastic strain) and elongation to failure (%) of a microstructure image."""
    inp = grain_inputs(img, params, pixel_um)
    eps, sig = tensile_curve(inp, params)
    ys = float(np.interp(0.002, eps, sig))
    hard = np.gradient(sig, eps)
    neck = np.nonzero((hard <= sig) & (eps > 0.002))[0]
    eps_u = eps[neck[0]] if len(neck) else eps[-1]
    el = float((np.exp(eps_u) - 1.0 + params.post_uniform) * 100.0)
    if not return_curve:
        return ys, el
    e_eng = np.exp(eps) - 1.0
    s_eng = sig / (1.0 + e_eng)
    cut = e_eng <= el / 100.0
    return ys, el, dict(true_strain=eps, true_stress=sig, eng_strain=e_eng[cut], eng_stress=s_eng[cut],
                        inputs=inp)
