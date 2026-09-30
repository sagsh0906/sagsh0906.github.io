"""Patch extraction and the three-step augmentation of the paper (Methods, "Microstructure database extension").

(i)   every 800 x 1000 map -> 8 x 10 = 80 non-overlapping 100 x 100 patches (2000 patches for 25 maps);
(ii)  6 geometric variants per patch: identity, horizontal / vertical flip, 90 / 180 / 270 deg rotation
      (lossless; 12 000 patches for the 2000 originals);
(iii) 5 degradations per geometric variant: Gaussian blur, affine distortion, black blocks, erosion, dilation
      (60 000 patches).  Degraded images are used only as *inputs*, the clean image being the target
      (restoration training); clean images are both input and target (reconstruction).
In total 72 000 (input, target) pairs, split 80 / 20 at the level of the original maps (no leakage).
Operators mirror the cv2 calls of the official ``Function.py`` (GaussianBlur 5x5 sigma 5, the same affine
point pairs, three 20 x 20 black blocks, 2 x 2 erosion / dilation), implemented with numpy / scipy.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch.utils.data import Dataset

GEOMETRIC = ('identity', 'hflip', 'vflip', 'rot90', 'rot180', 'rot270')
DEGRADATIONS = ('blur', 'affine', 'blocks', 'erode', 'dilate')


def crop_patches(img, patch=100):
    """(H, W, C) -> (H//patch * W//patch, patch, patch, C), row-major order (as in the paper's 8 x 10 grid)."""
    H, W = img.shape[:2]
    rows, cols = H // patch, W // patch
    p = img[:rows * patch, :cols * patch].reshape(rows, patch, cols, patch, -1).swapaxes(1, 2)
    return p.reshape(rows * cols, patch, patch, -1)


def stitch_patches(patches, rows=8, cols=10):
    n, h, w, c = patches.shape
    return patches.reshape(rows, cols, h, w, c).swapaxes(1, 2).reshape(rows * h, cols * w, c)


def geometric(p, kind):
    return {'identity': lambda a: a, 'hflip': lambda a: a[:, ::-1], 'vflip': lambda a: a[::-1],
            'rot90': lambda a: np.rot90(a, 1), 'rot180': lambda a: np.rot90(a, 2),
            'rot270': lambda a: np.rot90(a, 3)}[kind](p)


_G5 = np.exp(-((np.arange(5) - 2.0) ** 2) / (2 * 5.0 ** 2))
_G5 = np.outer(_G5, _G5) / np.outer(_G5, _G5).sum()


def degrade(p, kind, rng):
    """p: float (h, w, 3) in [0, 1]. Returns a degraded copy (information loss for restoration training)."""
    h, w = p.shape[:2]
    s = h / 100.0                                          # operators are specified for 100-px patches
    if kind == 'blur':
        return np.stack([ndimage.convolve(p[..., c], _G5, mode='reflect') for c in range(3)], -1)
    if kind == 'affine':
        # cv2.getAffineTransform([[50,50],[200,50],[50,200]] -> [[55,55],[190,50],[70,220]]), scaled to the patch
        src = np.array([[50, 50], [200, 50], [50, 200]], float) * s
        dst = np.array([[55, 55], [190, 50], [70, 220]], float) * s
        A = np.linalg.solve(np.c_[src, np.ones(3)], dst).T      # 2 x 3, dst = A @ [x, y, 1]
        M = np.linalg.inv(np.r_[A, [[0, 0, 1]]])[:2]              # output -> input (x, y)
        # ndimage works in (row, col) = (y, x)
        mat = np.array([[M[1, 1], M[1, 0]], [M[0, 1], M[0, 0]]])
        off = np.array([M[1, 2], M[0, 2]])
        return np.stack([ndimage.affine_transform(p[..., c], mat, off, order=1, mode='constant', cval=0.0)
                         for c in range(3)], -1)
    if kind == 'blocks':
        q = p.copy()
        b = max(int(round(20 * s)), 2)
        for _ in range(3):
            y, x = rng.integers(0, h - b), rng.integers(0, w - b)
            q[y:y + b, x:x + b] = 0.0
        return q
    if kind == 'erode':
        return ndimage.grey_erosion(p, size=(2, 2, 1))
    if kind == 'dilate':
        return ndimage.grey_dilation(p, size=(2, 2, 1))
    raise ValueError(kind)


class PatchDataset(Dataset):
    """(input, target) pairs from uint8 patches (N, h, w, 3).

    mode='paper' enumerates all N * 6 * (1 + 5) pairs (clean reconstruction + restoration), exactly the
    72 000-sample design for N = 2000; ``subsample`` draws a random subset of that index space per epoch.
    """

    def __init__(self, patches, img_size=None, degradations=True, seed=0):
        self.patches = patches
        self.img_size = img_size
        self.n_deg = len(DEGRADATIONS) if degradations else 0
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.patches) * len(GEOMETRIC) * (1 + self.n_deg)

    def _to_tensor(self, a):
        t = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).permute(2, 0, 1)
        if self.img_size is not None and self.img_size != t.shape[-1]:
            t = F.interpolate(t[None], size=self.img_size, mode='bilinear', antialias=True,
                              align_corners=False)[0].clamp(0, 1)
        return t

    def __getitem__(self, idx):
        n_var = 1 + self.n_deg
        pidx, rem = divmod(idx, len(GEOMETRIC) * n_var)
        gidx, didx = divmod(rem, n_var)
        clean = geometric(self.patches[pidx].astype(np.float32) / 255.0, GEOMETRIC[gidx])
        inp = clean if didx == 0 else degrade(np.ascontiguousarray(clean), DEGRADATIONS[didx - 1], self.rng)
        return self._to_tensor(inp), self._to_tensor(clean)


def to_tensor_batch(patches_uint8, img_size=None):
    t = torch.from_numpy(patches_uint8.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    if img_size is not None and img_size != t.shape[-1]:
        t = F.interpolate(t, size=img_size, mode='bilinear', antialias=True, align_corners=False).clamp(0, 1)
    return t


def to_numpy_images(t, size=None):
    """(B, 3, S, S) tensor in [0,1] -> (B, size, size, 3) float numpy (optionally resized back, e.g. 128 -> 100)."""
    if size is not None and size != t.shape[-1]:
        t = F.interpolate(t, size=size, mode='bilinear', antialias=True, align_corners=False)
    return t.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy()
