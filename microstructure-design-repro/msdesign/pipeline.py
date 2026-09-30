"""Helpers shared by the analysis scripts: load trained models, encode / reconstruct / generate patches in batches."""
import numpy as np
import torch

from .data import to_numpy_images, to_tensor_batch
from .models import Diffusion, PhysicsVAE


def load_vae(path):
    ck = torch.load(path, map_location='cpu')
    vae = PhysicsVAE(img_size=ck['img_size'])
    vae.load_state_dict(ck['state_dict'])
    return vae.eval(), ck


def load_ddpm(path):
    ck = torch.load(path, map_location='cpu')
    d = Diffusion()
    d.load_state_dict(ck['state_dict'])
    return d.eval()


@torch.no_grad()
def encode(vae, patches_uint8, batch=200):
    """Posterior means mu(x) of uint8 patches (N, h, w, 3) -> (N, z_dim)."""
    out = []
    for i in range(0, len(patches_uint8), batch):
        mu, _ = vae.encode(to_tensor_batch(patches_uint8[i:i + batch], vae.img_size))
        out.append(mu.numpy())
    return np.concatenate(out)


@torch.no_grad()
def decode(vae, z, ddpm=None, batch=100, seed=0, out_size=None):
    """Latents (N, z_dim) -> images (N, s, s, 3) float in [0,1]; optionally refined by the diffusion model."""
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(0, len(z), batch):
        x_v = vae.decode(torch.as_tensor(z[i:i + batch], dtype=torch.float32))
        if ddpm is not None:
            x_v = ddpm.sample(x_v, generator=g)
        out.append(to_numpy_images(x_v, out_size))
    return np.concatenate(out)


@torch.no_grad()
def reconstruct(vae, patches_uint8, ddpm=None, sample_latent=True, batch=100, seed=0):
    """Encode -> (sample) -> decode (-> refine). Returns float images at the patch resolution."""
    torch.manual_seed(seed)
    size = patches_uint8.shape[1]
    out_vae, out_ref = [], []
    g = torch.Generator().manual_seed(seed)
    for i in range(0, len(patches_uint8), batch):
        x = to_tensor_batch(patches_uint8[i:i + batch], vae.img_size)
        mu, logvar = vae.encode(x)
        z = vae.reparameterize(mu, logvar) if sample_latent else mu
        x_v = vae.decode(z)
        out_vae.append(to_numpy_images(x_v, size))
        if ddpm is not None:
            out_ref.append(to_numpy_images(ddpm.sample(x_v, generator=g), size))
    return np.concatenate(out_vae), (np.concatenate(out_ref) if ddpm is not None else None)


def quantize(img):
    """Round to 8 bit, as the images would be stored (PNG) before any physical analysis."""
    return (np.round(np.clip(img, 0, 1) * 255) / 255).astype(np.float32)
