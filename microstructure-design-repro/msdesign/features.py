"""Physically meaningful descriptors extracted from (original, reconstructed or generated) Euler-map images.

Grains are segmented from the black grain-boundary network that EBSD software draws on Euler maps, which is exactly
the interfacial information the edge loss is designed to preserve.  For every grain we compute the six features used
in the paper's fidelity analysis (Fig. 3): area, major / minor axis of the equivalent ellipse and the three mean
Euler angles.  Additional helpers give neighbour misorientations, kernel average misorientation (KAM) and simple
grain-size-distribution statistics used for the bimodality analysis (Fig. 5).
"""
import numpy as np
from scipy import ndimage
from skimage.measure import regionprops_table
from skimage.segmentation import expand_labels

from . import euler as E

PIXEL_UM = 0.5
FEATURE_NAMES = ['area', 'major_axis', 'minor_axis', 'euler1', 'euler2', 'euler3']
FEATURE_UNITS = ['um$^2$', 'um', 'um', 'deg', 'deg', 'deg']


def to_float_rgb(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def boundary_mask(img, dark_thresh=0.15):
    """Grain-boundary / non-indexed pixels = dark pixels (max channel below threshold)."""
    return to_float_rgb(img).max(axis=-1) < dark_thresh


def segment_grains(img, dark_thresh=0.15, min_px=3, fill_boundaries=False):
    """Connected components (4-connectivity) of non-boundary pixels. Label 0 = boundary/removed."""
    gb = boundary_mask(img, dark_thresh)
    labels, _ = ndimage.label(~gb)
    if min_px > 1:
        counts = np.bincount(labels.ravel())
        small = counts < min_px
        small[0] = False
        labels[small[labels]] = 0
        _, inv = np.unique(labels, return_inverse=True)
        labels = inv.reshape(labels.shape)
    if fill_boundaries:
        labels = expand_labels(labels, distance=3)
    return labels


def grain_table(img, labels=None, pixel_um=PIXEL_UM, **kw):
    """Per-grain table (dict of arrays) with the six physical features (um, um^2, degrees)."""
    img = to_float_rgb(img)
    if labels is None:
        labels = segment_grains(img, **kw)
    if labels.max() == 0:
        return {k: np.zeros(0) for k in FEATURE_NAMES + ['label']}
    eul_deg = img * E.EULER_RANGES_DEG
    props = regionprops_table(labels, intensity_image=eul_deg,
                              properties=('label', 'area', 'axis_major_length', 'axis_minor_length',
                                          'intensity_mean'))
    return {
        'label': props['label'],
        'area': props['area'] * pixel_um ** 2,
        'major_axis': props['axis_major_length'] * pixel_um,
        'minor_axis': props['axis_minor_length'] * pixel_um,
        'euler1': props['intensity_mean-0'],
        'euler2': props['intensity_mean-1'],
        'euler3': props['intensity_mean-2'],
    }


def mean_features(img, **kw):
    """Average of the six grain features for one image (the quantity plotted in Fig. 3i-n)."""
    t = grain_table(img, **kw)
    if len(t['area']) == 0:
        return np.full(len(FEATURE_NAMES), np.nan)
    return np.array([np.mean(t[k]) for k in FEATURE_NAMES])


def neighbour_misorientations(img, labels=None, **kw):
    """Misorientation angles (deg, cubic symmetry) between mean orientations of adjacent grains."""
    img = to_float_rgb(img)
    if labels is None:
        labels = expand_labels(segment_grains(img, **kw), distance=2)
    t = grain_table(img, labels=labels)
    if len(t['label']) < 2:
        return np.zeros(0)
    pairs = set()
    for a, b in ((labels[:, :-1], labels[:, 1:]), (labels[:-1, :], labels[1:, :])):
        m = (a != b) & (a > 0) & (b > 0)
        pairs.update(zip(np.minimum(a[m], b[m]).tolist(), np.maximum(a[m], b[m]).tolist()))
    if not pairs:
        return np.zeros(0)
    pairs = np.array(sorted(pairs))
    lut = np.zeros(labels.max() + 1, int)
    lut[t['label']] = np.arange(len(t['label']))
    eul = np.radians(np.stack([t['euler1'], t['euler2'], t['euler3']], 1))
    g = E.euler_to_matrix(eul)
    return E.misorientation_deg(g[lut[pairs[:, 0]]], g[lut[pairs[:, 1]]])


def kam_map(img, labels=None, step=1, max_deg=5.0, **kw):
    """Kernel average misorientation (deg) to the 4 neighbours at distance ``step`` inside the same grain."""
    img = to_float_rgb(img)
    if labels is None:
        labels = segment_grains(img, **kw)
    g = E.euler_to_matrix(E.rgb_to_euler(img))
    H, W = labels.shape
    acc = np.zeros((H, W))
    cnt = np.zeros((H, W))
    for dy, dx in ((0, step), (step, 0)):
        la, lb = labels[:H - dy, :W - dx], labels[dy:, dx:]
        same = (la == lb) & (la > 0)
        ga, gb_ = g[:H - dy, :W - dx][same], g[dy:, dx:][same]
        mis = E.misorientation_deg(ga, gb_)
        ok = mis < max_deg
        vals = np.where(ok, mis, 0.0)
        ya, xa = np.nonzero(same)
        np.add.at(acc, (ya, xa), vals)
        np.add.at(cnt, (ya, xa), ok)
        np.add.at(acc, (ya + dy, xa + dx), vals)
        np.add.at(cnt, (ya + dy, xa + dx), ok)
    with np.errstate(invalid='ignore'):
        kam = acc / cnt
    return kam / step, labels


def size_distribution_stats(areas_um2, fine_um2=None):
    """Summary statistics of a grain-area distribution, incl. Sarle's bimodality coefficient on log-areas."""
    a = np.asarray(areas_um2, float)
    a = a[a > 0]
    if len(a) < 4:
        return dict(n=len(a), mean=np.nan, median=np.nan, bimodality=np.nan, fine_fraction=np.nan)
    la = np.log10(a)
    n = len(la)
    m = la.mean()
    s = la.std()
    g1 = np.mean(((la - m) / s) ** 3)
    g2 = np.mean(((la - m) / s) ** 4) - 3.0
    bc = (g1 ** 2 + 1) / (g2 + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
    fine_um2 = np.median(a) / 4 if fine_um2 is None else fine_um2
    return dict(n=n, mean=float(a.mean()), median=float(np.median(a)), bimodality=float(bc),
                fine_fraction=float(np.mean(a <= fine_um2)))
