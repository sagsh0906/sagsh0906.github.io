"""Synthetic Inconel-625-like EBSD Euler maps for the 25 thermo-mechanical conditions of the paper.

The 25 experimental EBSD maps of the paper are not public (only 10 cropped 100x100 patches are released), so this
module produces physically motivated stand-ins that reproduce the *kind* of microstructures in the data set:

* hot rolling (0-50 % reduction) elongates grains along RD and stores dislocations, rendered as intragranular
  orientation gradients (smooth + short-range components -> KAM / GOS signal);
* static recrystallisation during heat treatment (JMAK kinetics, necklace nucleation at prior boundaries);
* recovery of the non-recrystallised matrix and grain growth after recrystallisation;
* annealing twins (Sigma-3, 60 deg about <111>) as straight bands inside grains;
* 1-px black grain boundaries, a few mis-indexed / non-indexed pixels, 800 x 1000 px at 0.5 um/px,
  colour-coded with the same Euler->RGB convention as the official data (see ``euler.py``).

Kinetic constants are rough, literature-order-of-magnitude values tuned so that the generated states follow the
qualitative trends of the measured data (``data/official/dataset.csv``); they are not fitted material constants.
"""
from dataclasses import dataclass, asdict

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from . import euler as E

R_GAS = 8.314
PIXEL_UM = 0.5
H_DEFAULT, W_DEFAULT = 800, 1000

# 25 orthogonal conditions (heat-treatment T [C], time [h], hot-rolling reduction [%]) = Supplementary Table 1
CONDITIONS = [
    (650, 1, 0), (750, 5, 0), (850, 10, 0), (950, 15, 0), (1050, 20, 0),
    (1050, 15, 20), (950, 10, 20), (850, 5, 20), (750, 1, 20), (650, 20, 20),
    (650, 15, 30), (750, 10, 30), (850, 20, 30), (950, 1, 30), (1050, 5, 30),
    (1050, 1, 40), (950, 5, 40), (850, 15, 40), (750, 20, 40), (650, 10, 40),
    (650, 5, 50), (750, 15, 50), (850, 1, 50), (950, 20, 50), (1050, 10, 50),
]


@dataclass
class ProcessState:
    eps: float          # von Mises equivalent rolling strain
    x_rx: float         # recrystallised area fraction
    d_def: float        # prior (deformed) grain size, um
    aspect: float       # aspect ratio of deformed grains (RD / ND)
    rho: float          # dislocation density left in the deformed matrix after recovery, m^-2
    d_rx: float         # recrystallised / annealed grain size, um
    twin_prob: float    # probability that an annealed grain contains twin bands


def processing_state(T_c, t_h, reduction_pct, d0=14.0):
    """Very small physical-metallurgy model: rolling -> recovery / JMAK recrystallisation -> grain growth."""
    T = T_c + 273.15
    r = reduction_pct / 100.0
    eps = 2.0 / np.sqrt(3.0) * -np.log(1.0 - r) if r > 0 else 0.0
    rho_ann = 1e12
    rho_def = rho_ann + 1.8e15 * eps ** 0.9
    # recovery of the deformed matrix (logarithmic-type decay)
    k_rec = 0.2 * np.exp(-200e3 / R_GAS * (1.0 / T - 1.0 / 1023.0))
    rho = rho_ann + (rho_def - rho_ann) * (1.0 + k_rec * t_h) ** -0.5
    # static recrystallisation, JMAK with Avrami exponent 1.5
    if eps > 0.05:
        t50 = 8.0 * (eps / 0.6) ** -2 * np.exp(450e3 / R_GAS * (1.0 / T - 1.0 / 1123.0))
        x_rx = 1.0 - np.exp(-0.693 * (t_h / t50) ** 1.5)
        t95 = t50 * (np.log(20.0) / 0.693) ** (1 / 1.5)
        d_rx0 = float(np.clip(7.0 * (eps / 0.6) ** -0.6, 3.0, 25.0))
        t_growth = max(0.0, t_h - t95)
    else:
        x_rx, d_rx0, t_growth = 1.0, d0, t_h   # un-rolled sheet: already recrystallised, only grain growth
    k_g = 172.0 * np.exp(-350e3 / R_GAS * (1.0 / T - 1.0 / 1323.0))   # um^2 / h
    d_rx = float(np.sqrt(d_rx0 ** 2 + k_g * t_growth))
    aspect = float(min((1.0 / (1.0 - r)) ** 2, 4.0)) if r > 0 else 1.0
    return ProcessState(eps=float(eps), x_rx=float(x_rx), d_def=d0, aspect=aspect, rho=float(rho),
                        d_rx=d_rx, twin_prob=0.4)


def _power_voronoi(seeds, weights, coords, k=8):
    """Assign coords to the seed minimising |x - s|^2 - w (power diagram), searched among k nearest seeds."""
    tree = cKDTree(seeds)
    k = min(k, len(seeds))
    d, idx = tree.query(coords, k=k, workers=-1)
    if k == 1:
        return idx
    score = d ** 2 - weights[idx]
    return idx[np.arange(len(coords)), np.argmin(score, axis=1)]


def _tessellate(n_grains, H, W, rng, aspect=1.0, mask=None, size_sigma=0.35):
    """Label map of an (anisotropic) power-Voronoi tessellation with log-normally dispersed cell sizes."""
    yy, xx = np.mgrid[0:H, 0:W]
    coords = np.stack([xx.ravel(), yy.ravel()], 1).astype(float)
    if mask is not None:
        cand = coords[mask.ravel()]
        seeds = cand[rng.choice(len(cand), size=min(n_grains, len(cand)), replace=False)]
    else:
        seeds = rng.random((n_grains, 2)) * [W, H]
    sx, sy = 1.0 / np.sqrt(aspect), np.sqrt(aspect)
    seeds_t = seeds * [sx, sy]
    spacing = np.sqrt(H * W / max(len(seeds), 1))
    radii = 0.5 * spacing * np.exp(size_sigma * rng.standard_normal(len(seeds)))
    # smooth random domain warp -> curved, non-polygonal boundaries
    corr = max(spacing / 3.0, 2.0)
    warp = ndimage.gaussian_filter(rng.standard_normal((2, H, W)), sigma=(0, corr, corr), mode='wrap')
    warp = warp / (warp.std(axis=(1, 2), keepdims=True) + 1e-12) * 0.18 * spacing
    coords_w = coords + warp.reshape(2, -1).T
    target = coords_w if mask is None else coords_w[mask.ravel()]
    lab = _power_voronoi(seeds_t, radii ** 2, target * [sx, sy])
    out = np.full(H * W, -1, dtype=np.int64)
    if mask is None:
        out[:] = lab
    else:
        out[mask.ravel()] = lab
    return out.reshape(H, W)


def _n_grains(d_um, area_px, pixel_um=PIXEL_UM):
    a_grain = np.pi * (d_um / pixel_um) ** 2 / 4.0
    return max(int(round(area_px / a_grain)), 2)


def _add_twins(labels, prob, rng, aspect=1.0, min_area=40):
    """Cut straight Sigma-3 twin bands through randomly chosen grains. Returns new labels and parent map."""
    labels = labels.copy()
    next_id = labels.max() + 1
    parent = {}
    slices = ndimage.find_objects(labels + 1)
    for lab, sl in enumerate(slices):
        if sl is None or rng.random() > prob:
            continue
        sub = labels[sl] == lab
        area = sub.sum()
        if area < min_area:
            continue
        yy, xx = np.nonzero(sub)
        theta = rng.uniform(0, np.pi)
        if aspect > 1:  # bands are rotated towards RD by the rolling deformation
            theta = np.arctan2(np.sin(theta), np.cos(theta) / aspect)
        s = xx * np.cos(theta) + yy * np.sin(theta)
        d_eq = np.sqrt(4 * area / np.pi)
        n_bands = rng.integers(1, 3)
        for _ in range(n_bands):
            c = rng.uniform(s.min(), s.max())
            w = max(2.0, rng.uniform(0.08, 0.25) * d_eq)
            band = np.abs(s - c) < w / 2
            if band.sum() < 4:
                continue
            gy, gx = yy[band] + sl[0].start, xx[band] + sl[1].start
            keep = labels[gy, gx] == lab
            labels[gy[keep], gx[keep]] = next_id
            parent[next_id] = lab
            next_id += 1
    return labels, parent


def _smooth_field(H, W, corr_px, rng):
    f = ndimage.gaussian_filter(rng.standard_normal((3, H, W)), sigma=(0, corr_px, corr_px), mode='wrap')
    return f / (f.std(axis=(1, 2), keepdims=True) + 1e-12)


def render(state: ProcessState, rng, H=H_DEFAULT, W=W_DEFAULT, pixel_um=PIXEL_UM, return_labels=False):
    """Render a synthetic EBSD Euler map (float RGB in [0,1], shape (H, W, 3)) with step size ``pixel_um``."""
    area = H * W
    px_scale = PIXEL_UM / pixel_um          # 1 at the paper's 0.5 um step
    x_rx = state.x_rx
    if x_rx >= 0.97:                       # fully recrystallised / annealed
        labels = _tessellate(_n_grains(state.d_rx, area, pixel_um), H, W, rng)
        rx_mask = np.ones((H, W), bool)
        deformed_ids = np.zeros(0, int)
    else:
        labels_def = _tessellate(_n_grains(state.d_def * np.sqrt(state.aspect), area, pixel_um), H, W, rng,
                                 aspect=state.aspect)
        labels = labels_def.copy()
        rx_mask = np.zeros((H, W), bool)
        if x_rx > 0.01:                    # necklace recrystallisation from the prior boundaries
            gb = np.zeros((H, W), bool)
            gb[:, :-1] |= labels_def[:, :-1] != labels_def[:, 1:]
            gb[:-1, :] |= labels_def[:-1, :] != labels_def[1:, :]
            dist = ndimage.distance_transform_edt(~gb)
            delta = np.quantile(dist, x_rx)
            rx_mask = dist <= delta
            n_rx = _n_grains(state.d_rx, rx_mask.sum(), pixel_um)
            rx_lab = _tessellate(n_rx, H, W, rng, mask=rx_mask)
            labels[rx_mask] = rx_lab[rx_mask] + labels_def.max() + 1
        deformed_ids = np.unique(labels[~rx_mask])
    # relabel consecutively, then add twins (annealing twins in both populations, sheared ones in deformed grains)
    _, labels = np.unique(labels, return_inverse=True)
    labels = labels.reshape(H, W)
    labels, twin_parent = _add_twins(labels, state.twin_prob, rng, aspect=state.aspect if x_rx < 0.97 else 1.0,
                                     min_area=max(int(40 * px_scale ** 2), 12))
    n_lab = max([labels.max()] + list(twin_parent)) + 1
    # orientations: random texture, twins are Sigma-3 related to their parent
    g = E.random_orientations(n_lab, rng)
    twin_axes = np.array([[1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1]], float)
    for child, par in twin_parent.items():
        ax = twin_axes[rng.integers(4)]
        g[child] = E.axis_angle_to_matrix(ax, np.pi / 3) @ g[par]
    euler = E.reduce_to_cubic_zone(g)[labels]                      # (H, W, 3)
    # deformed matrix: intragranular orientation gradients scaled with the stored dislocation density
    deformed = ~rx_mask
    if deformed.any() and state.eps > 0:
        b, u = 2.54e-10, pixel_um * 1e-6
        kam = np.clip(state.rho * b * u / 2.0, 0.0, np.radians(5.0))   # rad per pixel, rho = 2 theta / (u b)
        smooth = _smooth_field(H, W, 12.0 * px_scale, rng) * min(6.0 * kam, np.radians(10.0))
        rough = _smooth_field(H, W, 0.7, rng) * kam * 0.5
        rv = (smooth + rough).transpose(1, 2, 0)[deformed]
        gd = E.rotvec_to_matrix(rv) @ g[labels[deformed]]
        euler[deformed] = E.reduce_to_cubic_zone(gd)
    rgb = E.euler_to_rgb(euler)
    # 1-px black grain boundaries
    gb = np.zeros((H, W), bool)
    gb[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    gb[:-1, :] |= labels[:-1, :] != labels[1:, :]
    rgb[gb] = 0.0
    # EBSD noise: isolated mis-indexed pixels and a few non-indexed (black) specks, more in deformed regions
    n_mis = int(0.001 * area)
    iy, ix = rng.integers(0, H, n_mis), rng.integers(0, W, n_mis)
    rgb[iy, ix] = rng.random((n_mis, 3)) * [1.0, 0.6, 1.0]
    n_spk = int(area * 2e-4 * (1 + 4 * state.eps * (1 - x_rx)))
    iy, ix = rng.integers(0, H - 2, n_spk), rng.integers(0, W - 2, n_spk)
    for dy in range(2):
        for dx in range(2):
            rgb[iy + dy, ix + dx] = 0.0
    rgb = (np.round(rgb * 255) / 255).astype(np.float32)          # 8-bit quantisation like the PNG data
    if return_labels:
        return rgb, labels, rx_mask
    return rgb


def make_dataset(seed=0, H=H_DEFAULT, W=W_DEFAULT, pixel_um=PIXEL_UM, conditions=CONDITIONS):
    rng = np.random.default_rng(seed)
    images, states = [], []
    for (T, t, r) in conditions:
        st = processing_state(T, t, r)
        images.append(render(st, rng, H, W, pixel_um))
        states.append(asdict(st))
    return np.stack(images), states
