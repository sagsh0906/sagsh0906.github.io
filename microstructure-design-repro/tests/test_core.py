"""Fast sanity checks of the building blocks:  python3 -m pytest tests  (or  python3 tests/test_core.py)."""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, ROOT)

from msdesign import euler as E  # noqa: E402
from msdesign.data import GEOMETRIC, DEGRADATIONS, crop_patches, degrade, geometric, stitch_patches  # noqa: E402
from msdesign.inverse import DescriptorPCA, TreeEnsemble, average_hv, latent_descriptors  # noqa: E402
from msdesign.losses import edge_loss, pixel_loss, ssim  # noqa: E402
from msdesign.models import Diffusion, PhysicsVAE  # noqa: E402
from msdesign.nsga2 import nsga2, non_dominated_fronts  # noqa: E402


def test_orientation_math():
    rng = np.random.default_rng(0)
    g = E.random_orientations(4000, rng)
    e = E.reduce_to_cubic_zone(g)
    assert np.degrees(e[:, 1]).max() < 54.8 and np.degrees(e[:, 2]).max() < 90.0   # Channel-5 cubic zone
    assert E.misorientation_deg(g, E.euler_to_matrix(e)).max() < 1e-3                # symmetry-equivalent
    assert np.allclose(E.misorientation_deg(g[:10], E.sigma3_twin(g[:10])), 60.0)
    m = E.misorientation_deg(g[:2000], g[2000:])
    assert 39.0 < m.mean() < 42.0 and m.max() < 62.9                                 # Mackenzie distribution
    rgb = E.euler_to_rgb(e)
    assert np.allclose(E.rgb_to_euler(rgb), e)


def test_losses():
    x = torch.rand(2, 3, 32, 32)
    assert abs(ssim(x, x).item() - 1.0) < 1e-5
    assert pixel_loss(x, x).item() == 0.0 and edge_loss(x, x).item() == 0.0
    assert ssim(x, torch.rand_like(x)).item() < 0.2


def test_models_and_official_weights():
    vae = PhysicsVAE(img_size=64)
    out, mu, logvar = vae(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 3, 64, 64) and mu.shape == (2, 128)
    path = os.path.join(ROOT, '..', '..', 'nwpuai4msegroup', 'microstructures_design', 'model_save', 'diffusVAE.pt')
    if os.path.exists(path):
        Diffusion().load_state_dict(torch.load(path, map_location='cpu'), strict=True)


def test_data_ops():
    img = np.random.default_rng(0).integers(0, 255, (160, 200, 3), dtype=np.uint8)
    p = crop_patches(img, 20)
    assert p.shape == (80, 20, 20, 3) and np.array_equal(stitch_patches(p, 8, 10), img)
    f = p[0].astype(np.float32) / 255
    for g in GEOMETRIC:
        assert geometric(f, g).shape == f.shape
    for d in DEGRADATIONS:
        assert degrade(f, d, np.random.default_rng(0)).shape == f.shape


def test_nsga2_zdt1():
    def zdt1(X):
        f1 = X[:, 0]
        g = 1 + 9 * X[:, 1:].mean(1)
        return np.stack([f1, g * (1 - np.sqrt(f1 / g))], 1)
    res = nsga2(zdt1, np.zeros(6), np.ones(6), pop_size=20, n_gen=200, seed=1)
    F = res['F']
    assert np.all(np.abs(F[:, 1] - (1 - np.sqrt(F[:, 0]))) < 0.15)


def test_hv_and_fronts():
    Y = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.5, 1.5]])
    assert set(non_dominated_fronts(-Y)[0].tolist()) == {0, 1, 2}
    assert np.isclose(average_hv(Y, np.zeros(2)), (3 + 4 + 3) / 3)


def test_official_surrogates_reproduce_paper():
    d = os.path.join(ROOT, 'data', 'official')
    z = np.genfromtxt(os.path.join(d, 'z_i.csv'), delimiter=',', skip_header=1)[:, 1:]
    Y = np.genfromtxt(os.path.join(d, 'dataset.csv'), delimiter=',', skip_header=1)[:, 4:6]
    zm, zs = latent_descriptors(z)
    X = DescriptorPCA(3, 3).fit(zm, zs).transform(zm, zs)
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    for name, col, seed, target in (('GBR_ys', 0, 75, 0.961), ('GBR_el', 1, 98, 0.922)):
        m = TreeEnsemble(os.path.join(d, 'gbr_official_trees.npz'), name)
        _, X_te, _, y_te = train_test_split(X, Y[:, col], test_size=0.2, random_state=seed)
        assert abs(r2_score(y_te, m.predict(X_te)) - target) < 2e-3


if __name__ == '__main__':
    for k, v in list(globals().items()):
        if k.startswith('test_'):
            v()
            print('ok', k)
