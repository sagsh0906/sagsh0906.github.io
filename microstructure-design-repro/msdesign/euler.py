"""Orientation utilities: Bunge Euler angles, cubic symmetry and the EBSD Euler-map <-> RGB mapping.

The paper stores each EBSD map as an RGB image through a one-to-one mapping of the three Euler angles to the
three colour channels (Channel 5 "Euler colouring" for cubic crystals).  Inspecting the example patches of the
official repository gives R in [0, 255], G in [0, ~155] and B in [0, 255], i.e.

    R = phi1 / 360 deg,   G = Phi / 90 deg,   B = phi2 / 90 deg

with the orientation reduced to the cubic zone phi2 in [0, 90), Phi <= ~54.7 deg (the 3 copies of the cubic
fundamental zone inside phi2, Phi in [0, 90] are resolved by taking the smallest Phi).  All helpers below work
with numpy arrays and are vectorised over leading dimensions.
"""
import itertools

import numpy as np

EULER_RANGES_DEG = np.array([360.0, 90.0, 90.0])  # per-channel full scale of the RGB mapping


def cubic_symmetry_matrices():
    """The 24 proper rotations of the cubic point group m-3m (signed permutation matrices, det = +1)."""
    ops = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            m = np.zeros((3, 3))
            for i, (p, s) in enumerate(zip(perm, signs)):
                m[i, p] = s
            if np.isclose(np.linalg.det(m), 1.0):
                ops.append(m)
    return np.stack(ops)


CUBIC_SYM = cubic_symmetry_matrices()


def euler_to_matrix(euler):
    """Bunge (phi1, Phi, phi2) in radians -> orientation matrix g (crystal <- sample). euler: (..., 3)."""
    p1, P, p2 = euler[..., 0], euler[..., 1], euler[..., 2]
    c1, s1, c, s, c2, s2 = np.cos(p1), np.sin(p1), np.cos(P), np.sin(P), np.cos(p2), np.sin(p2)
    g = np.empty(euler.shape[:-1] + (3, 3))
    g[..., 0, 0] = c1 * c2 - s1 * s2 * c
    g[..., 0, 1] = s1 * c2 + c1 * s2 * c
    g[..., 0, 2] = s2 * s
    g[..., 1, 0] = -c1 * s2 - s1 * c2 * c
    g[..., 1, 1] = -s1 * s2 + c1 * c2 * c
    g[..., 1, 2] = c2 * s
    g[..., 2, 0] = s1 * s
    g[..., 2, 1] = -c1 * s
    g[..., 2, 2] = c
    return g


def matrix_to_euler(g):
    """Orientation matrix -> Bunge Euler angles in radians, phi1, phi2 in [0, 2pi), Phi in [0, pi]."""
    g33 = np.clip(g[..., 2, 2], -1.0, 1.0)
    P = np.arccos(g33)
    sP = np.sin(P)
    regular = sP > 1e-6
    p1 = np.where(regular, np.arctan2(g[..., 2, 0], -g[..., 2, 1]), np.arctan2(g[..., 0, 1], g[..., 0, 0]))
    p2 = np.where(regular, np.arctan2(g[..., 0, 2], g[..., 1, 2]), 0.0)
    return np.stack([np.mod(p1, 2 * np.pi), P, np.mod(p2, 2 * np.pi)], axis=-1)


def axis_angle_to_matrix(axis, angle):
    """Rodrigues formula. axis (..., 3) (need not be normalised), angle (...) in radians."""
    axis = np.asarray(axis, dtype=float)
    n = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), 1e-12)
    x, y, z = n[..., 0], n[..., 1], n[..., 2]
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    R = np.empty(np.broadcast(x, angle).shape + (3, 3))
    R[..., 0, 0] = c + x * x * C
    R[..., 0, 1] = x * y * C - z * s
    R[..., 0, 2] = x * z * C + y * s
    R[..., 1, 0] = y * x * C + z * s
    R[..., 1, 1] = c + y * y * C
    R[..., 1, 2] = y * z * C - x * s
    R[..., 2, 0] = z * x * C - y * s
    R[..., 2, 1] = z * y * C + x * s
    R[..., 2, 2] = c + z * z * C
    return R


def rotvec_to_matrix(rotvec):
    """Rotation vector (axis * angle) -> matrix."""
    angle = np.linalg.norm(rotvec, axis=-1)
    return axis_angle_to_matrix(np.where(angle[..., None] > 1e-12, rotvec, [1.0, 0.0, 0.0]), angle)


def random_orientations(n, rng):
    """Uniformly distributed random orientations (via uniform random unit quaternions)."""
    u1, u2, u3 = rng.random((3, n))
    q = np.stack([np.sqrt(1 - u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1) * np.cos(2 * np.pi * u2),
                  np.sqrt(u1) * np.sin(2 * np.pi * u3), np.sqrt(u1) * np.cos(2 * np.pi * u3)], axis=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], axis=-2)


def sigma3_twin(g):
    """Sigma-3 (60 deg about <111>) twin variant of orientation(s) g, rotation applied in the crystal frame."""
    R = axis_angle_to_matrix(np.array([1.0, 1.0, 1.0]), np.pi / 3)
    return R @ g


def reduce_to_cubic_zone(g, chunk=200_000):
    """Pick, among the 24 cubic equivalents of g, the Euler triplet with phi2 in [0, 90) deg and minimal Phi.

    Returns Bunge angles in radians with phi1 in [0, 2pi), Phi in [0, ~0.955], phi2 in [0, pi/2).
    """
    flat = g.reshape(-1, 3, 3)
    out = np.empty((flat.shape[0], 3))
    for start in range(0, flat.shape[0], chunk):
        gg = flat[start:start + chunk]
        eq = np.einsum('kij,njl->nkil', CUBIC_SYM, gg)            # (n, 24, 3, 3)
        e = matrix_to_euler(eq)                                    # (n, 24, 3)
        ok = (e[..., 2] < np.pi / 2 - 1e-9) & (e[..., 1] <= np.pi / 2 + 1e-9)
        score = np.where(ok, e[..., 1], np.inf)
        best = np.argmin(score, axis=1)
        out[start:start + chunk] = e[np.arange(len(gg)), best]
    return out.reshape(g.shape[:-2] + (3,))


def euler_to_rgb(euler_rad):
    """Reduced Bunge Euler angles (radians) -> float RGB in [0, 1] (Channel 5 cubic convention)."""
    rgb = np.degrees(euler_rad) / EULER_RANGES_DEG
    return np.clip(rgb, 0.0, 1.0)


def rgb_to_euler(rgb):
    """Float RGB in [0, 1] -> Bunge Euler angles in radians (inverse of euler_to_rgb)."""
    return np.radians(np.asarray(rgb, dtype=float) * EULER_RANGES_DEG)


def misorientation_deg(g_a, g_b):
    """Cubic-symmetry-reduced misorientation angle (degrees) between orientation arrays g_a, g_b (..., 3, 3)."""
    m = np.einsum('...ij,...kj->...ik', g_a, g_b)                  # g_a g_b^T
    tr = np.einsum('sij,...ji->...s', CUBIC_SYM, m)                # tr(S m) for the 24 operators
    cos = np.clip((tr.max(axis=-1) - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def schmid_factor_fcc(g, load_dir=(1.0, 0.0, 0.0)):
    """Maximum Schmid factor of the 12 {111}<110> systems for uniaxial load along a sample direction."""
    normals = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [-1, 1, 1], [-1, 1, 1], [-1, 1, 1],
                        [1, -1, 1], [1, -1, 1], [1, -1, 1], [1, 1, -1], [1, 1, -1], [1, 1, -1]], float)
    dirs = np.array([[0, 1, -1], [1, 0, -1], [1, -1, 0], [0, 1, -1], [1, 0, 1], [1, 1, 0],
                     [0, 1, 1], [1, 0, -1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, -1, 0]], float)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    load_c = np.einsum('...ij,j->...i', g, np.asarray(load_dir, float))  # load direction in crystal frame
    m = np.abs((load_c @ normals.T) * (load_c @ dirs.T))
    return m.max(axis=-1)
